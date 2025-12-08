import argparse
import json
import math
import os
import sys
from tqdm import tqdm
from glob import glob
from collections import defaultdict
import pickle

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from qwen_vl_utils import process_vision_info

from pycocotools import mask as cocomask
import re

from evaluation.inf_utils import *
from evaluation.llm_prompting_utils import obj_cls_predict_prompting, background_caption_prompting, obj_choice_prompting
import evaluation.llm_prompting_utils as llm_prompting_utils
from model.modeling_qwen2_5_vl_custom import Qwen2_5_VLForConditionalGenerationCustom
from model.modeling_qwen2_vl_custom import Qwen2VLForConditionalGenerationCustom
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
    max_tokens_per_video, resize_scale = args.attn_max_token, args.attn_resize_scale
    llm_prompting_utils.MAX_BATCH = args.attn_max_batch
    prompt_obj_ver = args.prompt_obj_ver
    prompt_bg_ver = args.prompt_bg_ver
    prompt_obj_cnt_ver = args.prompt_obj_cnt_ver
    resized_height, resized_width = -1, -1

    # ------------------ initialize MLLM and SAM models -------------------------
    if "Qwen2.5-VL" in args.version:
        model_class = Qwen2_5_VLForConditionalGenerationCustom
    elif "Qwen2-VL" in args.version:
        model_class = Qwen2VLForConditionalGenerationCustom
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
        args.version,
        cache_dir=None,
        padding_side="right",
        use_fast=False,
    )

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
        size_per_tok = 28
        ## object class prediction prompting
        obj_cls_results_video = obj_cls_predict_prompting(model, processor, ref_query, frame_file_path, video_flag=True, tokens_per_frame=max_tokens_per_video, prompt_ver=prompt_obj_ver)
        grid_t, grid_h, grid_w = obj_cls_results_video['visual_grid_thw'][0].tolist()
        grid_h, grid_w = grid_h // 2, grid_w // 2 # feature map size
        if args.attn_max_token_per_image == -1:
            max_tokens_per_image = max_tokens_per_video * resize_scale * resize_scale 
            resized_height, resized_width = grid_h * size_per_tok * resize_scale, grid_w * size_per_tok * resize_scale
        else:
            max_tokens_per_image = args.attn_max_token_per_image
        obj_cls_results_frame = obj_cls_predict_prompting(model, processor, ref_query, frame_file_path, video_flag=False, tokens_per_frame=max_tokens_per_image, resized_height=resized_height, resized_width=resized_width, prompt_ver=prompt_obj_ver)
        
        ## object class selection prompting
        obj_list = [*obj_cls_results_video['obj_cls_list'], *obj_cls_results_frame['obj_cls_list']]
        obj_name = obj_choice_prompting(model, processor, ref_query, obj_list, prompt_ver=prompt_obj_cnt_ver)

        ## background captioning prompting
        bg_caption_results_video = background_caption_prompting(model, processor, obj_name, frame_file_path, video_flag=True, tokens_per_frame=max_tokens_per_video, prompt_ver=prompt_bg_ver, ref_query=ref_query)
        grid_t, grid_h, grid_w = bg_caption_results_video['visual_grid_thw'][0].tolist()
        grid_h, grid_w = grid_h // 2, grid_w // 2 # feature map size
        if args.attn_max_token_per_image == -1:
            max_tokens_per_image = max_tokens_per_video * resize_scale * resize_scale 
            resized_height, resized_width = grid_h * size_per_tok * resize_scale, grid_w * size_per_tok * resize_scale
        else:
            max_tokens_per_image = args.attn_max_token_per_image
        bg_caption_results_frame = background_caption_prompting(model, processor, obj_name, frame_file_path, video_flag=False, tokens_per_frame=max_tokens_per_image, resized_height=resized_height, resized_width=resized_width, prompt_ver=prompt_bg_ver, ref_query=ref_query)
        
        ## save results
        llm_results = {
            "obj_cls_results_video": obj_cls_results_video,
            "obj_cls_results_frame": obj_cls_results_frame,
            "bg_caption_results_video": bg_caption_results_video,
            "bg_caption_results_frame": bg_caption_results_frame,
            "key_frame_indices": key_frame_indices,
        }

        if attn_save_filepath:
            if not os.path.exists(attn_save_path):
                os.makedirs(attn_save_path, exist_ok=True)
            torch.save(llm_results, attn_save_filepath)
            torch.cuda.empty_cache()

        progress_bar.update(1)


if __name__ == "__main__":
    main(sys.argv[1:])
