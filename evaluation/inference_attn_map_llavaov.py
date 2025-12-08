import argparse
import math
import json
import os
import sys
from tqdm import tqdm
from glob import glob
from collections import defaultdict
import pickle
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers

from evaluation.inf_utils import *
from evaluation.llm_prompting_utils import obj_cls_predict_prompting_llavaov, background_caption_prompting_llavaov, obj_choice_prompting_llavaov
import evaluation.llm_prompting_utils as llm_prompting_utils

from model.modeling_llavaonevision_custom import (
    LlavaOnevisionForConditionalGenerationCustom, llava_imageprocessor_preprocess,
    llavaov_processor_call, llavaov_get_number_of_image_features, llavaov_get_image_patches,
    llavaov_get_unpadded_features, llavaov_expand_image_tokens,
)

from transformers.models.llava_onevision.processing_llava_onevision import LlavaOnevisionProcessor
from transformers.models.llava_onevision.image_processing_llava_onevision import LlavaOnevisionImageProcessor
transformers.logging.set_verbosity_error()  # hides transformers warnings


def main(args):
    args = parse_args(args)
    print(f"Arguments: {args}")
    tgt_dataset, testset = args.dataset, args.testset
    subset_num, subset_idx = args.subset_num, args.subset_idx
    dtype = torch.bfloat16
    
    attn_save_path = args.attn_save_path
    num_input_frames = args.num_input_frames
    attn_layer_idx_start = args.attn_layer_idx_start
    attn_layer_idx_end = args.attn_layer_idx_end
    llm_prompting_utils.MAX_BATCH = args.attn_max_batch

    # ------------------ initialize MLLM and SAM models -------------------------
    model_path = args.version
    if "llava-video" in model_path.lower() or "llava-onevision" in model_path.lower():
        model_class = LlavaOnevisionForConditionalGenerationCustom
        LlavaOnevisionProcessor.__call__ = llavaov_processor_call
        LlavaOnevisionProcessor._get_number_of_features = llavaov_get_number_of_image_features
        LlavaOnevisionProcessor._get_unpadded_features = llavaov_get_unpadded_features
        LlavaOnevisionProcessor._expand_image_tokens = llavaov_expand_image_tokens
        LlavaOnevisionImageProcessor.get_image_patches = llavaov_get_image_patches
        LlavaOnevisionImageProcessor.preprocess = llava_imageprocessor_preprocess
    else:
        raise NotImplementedError(f"{args.version} is not yet implemented")

    model = model_class.from_pretrained(
        args.version,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model = model.to(dtype).cuda()
    model.eval()
    processor = transformers.AutoProcessor.from_pretrained(
        model_path, cache_dir=None, padding_side="right", use_fast=False,
    )
    
    ## suppress image & video separator tokens (to keep grid shape for easy implementatino)
    model.model.use_newline = False
    processor.use_newline = False

    attn_layer_idx_end = attn_layer_idx_end if attn_layer_idx_end > 0 else len(model.model.language_model.layers)
    model.model.language_model.attn_layer_idx_start = attn_layer_idx_start
    model.model.language_model.attn_layer_idx_end = attn_layer_idx_end
    for i in range(len(model.model.language_model.layers)):
        model.model.language_model.layers[i].self_attn.attn_layer_flag = True if i >= attn_layer_idx_start and i < attn_layer_idx_end else False

    # ------------------ initialize dataset -------------------------
    video_folder, meta_exp, mask_dict = get_dataset(tgt_dataset, testset)

    # ------------------ split into subsets for each GPU -------------------------
    job_list_subset = split_eval_data(video_folder, meta_exp, subset_num, subset_idx, attn_save_path=attn_save_path)
    total_infer = len(job_list_subset)
    progress_bar = tqdm(total=total_infer, desc="Progress {}".format(subset_idx))

    # ------------------ start processing -------------------------

    for vid_id, exp_id in job_list_subset:
        frame_len = len(os.listdir(os.path.join(video_folder, vid_id)))
        vid_name = meta_exp[vid_id]["vid_id"] if "vid_id" in meta_exp[vid_id] else vid_id
        attn_save_filepath = os.path.join(attn_save_path, f"{vid_name}_{exp_id}.pt") if attn_save_path else None
        curr_expr = meta_exp[vid_id]["expressions"][exp_id]

        if os.path.exists(attn_save_filepath):
            print(f"Skip {vid_name}_{exp_id}")
            progress_bar.update(1)
            continue
        
        # preprocess text prompt
        ref_query = curr_expr["exp"]
        if not ref_query.endswith("?") and not ref_query.endswith("."):
            ref_query += "."
        
        ## frame sampling
        # uniform sampled frames
        if frame_len <= num_input_frames:
            key_frame_indices = list(range(frame_len))
        else:
            # uniform sampling
            key_frame_indices = np.linspace(0, frame_len - 1, num_input_frames).astype(int).tolist()

        image_list_ori, frame_file_path, original_size_list, image_file_list = load_and_process_video(video_folder, vid_id, dtype, key_frame_indices)

        # ---------------------------------------------------------------
        ## Step 1. LLM prompting to obtain attention maps
        
        ## object class prediction prompting
        obj_cls_results_video = obj_cls_predict_prompting_llavaov(model, processor, ref_query, frame_file_path, video_flag=True)
        obj_cls_results_frame = obj_cls_predict_prompting_llavaov(model, processor, ref_query, frame_file_path, video_flag=False)
        
        ## object class selection prompting
        obj_list = [*obj_cls_results_video['obj_cls_list'], *obj_cls_results_frame['obj_cls_list']]
        obj_name = obj_choice_prompting_llavaov(model, processor, ref_query, obj_list)

        ## background captioning prompting
        bg_caption_results_video = background_caption_prompting_llavaov(model, processor, obj_name, frame_file_path, video_flag=True)
        bg_caption_results_frame = background_caption_prompting_llavaov(model, processor, obj_name, frame_file_path, video_flag=False)
        llm_results = {
            "obj_cls_results_video": obj_cls_results_video,
            "obj_cls_results_frame": obj_cls_results_frame,
            "bg_caption_results_video": bg_caption_results_video,
            "bg_caption_results_frame": bg_caption_results_frame,
            "key_frame_indices": key_frame_indices,
        }
        llm_results

        if attn_save_filepath:
            if not os.path.exists(attn_save_path):
                os.makedirs(attn_save_path, exist_ok=True)
            torch.save(llm_results, attn_save_filepath)
            torch.cuda.empty_cache()

        progress_bar.update(1)


if __name__ == "__main__":
    main(sys.argv[1:])
