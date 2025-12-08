import argparse
import json
import os
import sys
from tqdm import tqdm
from glob import glob
from collections import defaultdict
import pickle
import einops
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers

# from model.sam2_video_predictor_custom import build_sam2_video_predictor

from pycocotools import mask as cocomask
import re

from evaluation.inf_utils import *

transformers.logging.set_verbosity_error()  # hides transformers warnings

def main(args):
    args = parse_args(args)
    print(f"Arguments: {args}")
    tgt_dataset, testset = args.dataset, args.testset
    subset_num, subset_idx = args.subset_num, args.subset_idx
    dtype = torch.float16 if "AWQ" in args.version else torch.bfloat16
    
    attn_save_path = args.attn_save_path
    use_saved_attn = args.use_saved_attn
    num_input_frames = args.num_input_frames

    llava_flag = "llava-ov" in args.attn_save_path
    internvl_flag = "intern-vl" in args.attn_save_path

    # ------------------ initialize MLLM and SAM models -------------------------
    if not args.use_saved_attn:
        raise NotImplementedError("Please use saved attention maps for evaluation.")

    # ------------------ initialize dataset -------------------------
    vis_save_path = args.vis_save_path
    video_folder, meta_exp, mask_dict = get_dataset(tgt_dataset, testset)

    # ------------------ split into subsets for each GPU -------------------------
    job_list_subset = split_eval_data(video_folder, meta_exp, subset_num, subset_idx, vis_save_path)
    total_infer = len(job_list_subset)
    progress_bar = tqdm(total=total_infer, desc="Progress {}".format(subset_idx))

    # ------------------ start processing -------------------------
    for vid_id, exp_id in job_list_subset:
        curr_mask_save_dir = os.path.join(vis_save_path, vid_id, exp_id)
        keypoint_save_dir = os.path.join(curr_mask_save_dir, "keypoints.pt")
        frame_len = len(os.listdir(os.path.join(video_folder, vid_id)))
        vid_name = meta_exp[vid_id]["vid_id"] if "vid_id" in meta_exp[vid_id] else vid_id
        attn_save_filepath = os.path.join(attn_save_path, f"{vid_name}_{exp_id}.pt") if attn_save_path else None
        curr_expr = meta_exp[vid_id]["expressions"][exp_id]
        
        if os.path.exists(keypoint_save_dir):
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

        # image_list_ori, frame_file_path, original_size_list, image_file_list = load_and_process_video(video_folder, vid_id, dtype, key_frame_indices)
        
        # ---------------------------------------------------------------
        ## Step 1. LLM prompting to obtain attention maps
        if use_saved_attn:
            llm_results = torch.load(attn_save_filepath, weights_only=True)
        else:
            raise NotImplementedError("Please use saved attention maps for evaluation.")

        ## Step 2. Fuse attention maps
        ## smooth the attention map
        if llava_flag:
            attn_map_obj_video = get_smooth_attn_map_thw_llava(llm_results, tgt_key="obj_cls_results_video", smooth_size=3)
            attn_map_obj_frame = get_smooth_attn_map_thw_llava(llm_results, tgt_key="obj_cls_results_frame", smooth_size=7)
            attn_map_bg_video = get_smooth_attn_map_thw_llava(llm_results, tgt_key="bg_caption_results_video", smooth_size=3)
            attn_map_bg_frame = get_smooth_attn_map_thw_llava(llm_results, tgt_key="bg_caption_results_frame", smooth_size=7)
        elif internvl_flag:
            attn_map_obj_video = get_smooth_attn_map_thw_internvl(llm_results, tgt_key="obj_cls_results_video", smooth_size=3)
            attn_map_obj_frame = get_smooth_attn_map_thw_internvl(llm_results, tgt_key="obj_cls_results_frame", smooth_size=7)
            attn_map_bg_video = get_smooth_attn_map_thw_internvl(llm_results, tgt_key="bg_caption_results_video", smooth_size=3)
            attn_map_bg_frame = get_smooth_attn_map_thw_internvl(llm_results, tgt_key="bg_caption_results_frame", smooth_size=7)
        else:
            attn_map_obj_video = get_smooth_attn_map_thw(llm_results, tgt_key="obj_cls_results_video", smooth_size=3)
            attn_map_obj_frame = get_smooth_attn_map_thw(llm_results, tgt_key="obj_cls_results_frame", smooth_size=7)
            attn_map_bg_video = get_smooth_attn_map_thw(llm_results, tgt_key="bg_caption_results_video", smooth_size=3)
            attn_map_bg_frame = get_smooth_attn_map_thw(llm_results, tgt_key="bg_caption_results_frame", smooth_size=7)

        ## object - background contrasting
        attn_map_obj_bg_video = torch.clamp(attn_map_obj_video - attn_map_bg_video, min=0.0, max=1.0)
        attn_map_obj_bg_frame = torch.clamp(attn_map_obj_frame - attn_map_bg_frame, min=0.0, max=1.0)

        norm_attn_map_obj_bg_video = normalize_attn_map_thw(attn_map_obj_bg_video, frame_wise_norm=False)
        norm_attn_map_obj_bg_frame = normalize_attn_map_thw(attn_map_obj_bg_frame, frame_wise_norm=True)
        norm_attn_map_video, norm_attn_map_frame = norm_attn_map_obj_bg_video, norm_attn_map_obj_bg_frame

        ## match video-frame input shape (qwen2vl downsamples in temporal-axis).
        ## When odd number frames, remove extra padded frame
        if len(norm_attn_map_video) != len(norm_attn_map_frame):
            norm_attn_map_video = norm_attn_map_video.repeat_interleave(2, dim=0)
        if len(norm_attn_map_video) != len(norm_attn_map_frame):
            norm_attn_map_video = norm_attn_map_video[:-1]

        ## video - frame fusion
        _, img_h, img_w = norm_attn_map_frame.shape
        _norm_attn_map_video = F.interpolate(norm_attn_map_video.unsqueeze(1), size=(img_h, img_w), mode='bilinear').squeeze(1)            
        fused_attn_map = (_norm_attn_map_video + norm_attn_map_frame) / 2.0
        fused_attn_map = normalize_attn_map_thw(fused_attn_map, frame_wise_norm=False)

        assert len(key_frame_indices) == len(fused_attn_map), "Key frame indices length does not match attention map length."

        ## add batch-axis
        video_heads_attn = fused_attn_map.unsqueeze(0)
        video_heads_attn = video_heads_attn.cuda() if video_heads_attn.device == torch.device("cpu") else video_heads_attn
        
        ## make mask saving directory
        if not os.path.exists(curr_mask_save_dir):
            os.makedirs(curr_mask_save_dir, exist_ok=True)
        
        ## Step 3. Extract keypoints
        if len(meta_exp[vid_id]['frames']) != frame_len:
            print(f"[Frame-Mask Mismatch] {vid_id} Video")

        # if vid_name == "353_-5sqmCF8OFU":
        #     video_heads_attn

        K = 100
        x = video_heads_attn[-1]
        T, H, W = x.shape
        vals, flat_idxs = torch.topk(x.view(-1), K)
        t_idxs = flat_idxs // (H * W)
        yx_idxs = flat_idxs % (H * W)
        y_idxs = yx_idxs // W
        x_idxs = yx_idxs % W
        _y_idxs = (y_idxs + 0.5) / H
        _x_idxs = (x_idxs + 0.5) / W
        _key_frame_indices = torch.tensor(key_frame_indices).to(t_idxs)
        tyx_val_idxs = torch.stack([_key_frame_indices[t_idxs], _y_idxs, _x_idxs, vals], dim=1).cpu() # [K, 4]
        torch.save(tyx_val_idxs, keypoint_save_dir)
        
        progress_bar.update(1)


if __name__ == "__main__":
    main(sys.argv[1:])
