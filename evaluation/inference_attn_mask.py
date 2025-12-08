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

from model.sam2_video_predictor_custom import build_sam2_video_predictor

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

    predict_attn_mask = args.predict_attn_mask
    predict_sam_mask = args.predict_sam_mask
    sam_mask_score_thresh, sam_point_threshold = args.final_score_threshold, args.point_threshold
    llava_flag = "llava-ov" in args.attn_save_path
    internvl_flag = "intern-vl" in args.attn_save_path

    # ------------------ initialize MLLM and SAM models -------------------------
    if not args.use_saved_attn:
        raise NotImplementedError("Please use saved attention maps for evaluation.")

    if predict_sam_mask:
        predictor = build_sam2_video_predictor(args.vision_pretrained_config, args.vision_pretrained, device=torch.device("cuda"))

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
        frame_len = len(os.listdir(os.path.join(video_folder, vid_id)))
        vid_name = meta_exp[vid_id]["vid_id"] if "vid_id" in meta_exp[vid_id] else vid_id
        attn_save_filepath = os.path.join(attn_save_path, f"{vid_name}_{exp_id}.pt") if attn_save_path else None
        curr_expr = meta_exp[vid_id]["expressions"][exp_id]
        
        if os.path.exists(curr_mask_save_dir): # obtain masks
            sam_mask_done_flag = predict_sam_mask and len(os.listdir(curr_mask_save_dir)) == frame_len
            attn_mask_frame_len = frame_len if frame_len < num_input_frames else num_input_frames
            attn_mask_done_flag = predict_attn_mask and len(os.listdir(curr_mask_save_dir)) == attn_mask_frame_len
            if sam_mask_done_flag or attn_mask_done_flag:
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
        if use_saved_attn:
            llm_results = torch.load(attn_save_filepath, weights_only=True)
        else:
            raise NotImplementedError("Please use saved attention maps for evaluation.")

        if predict_attn_mask or predict_sam_mask:
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
        
        if predict_attn_mask:
            ## Step 3. Obtain binary mask from attention weights
            assert video_heads_attn.shape[1] == len(key_frame_indices), (
                f"Attention map length {video_heads_attn.shape[1]} does not match key frame indices length {len(key_frame_indices)}."
            )
            pred_masks = get_mask_preds_directly_from_attn(video_heads_attn, original_size_list)
            if pred_masks is None:
                print(f"No predictions for video {vid_id}, expression {exp_id}.")
                continue
            
            ## opt1: save all frames
            save_frame_indices = key_frame_indices
            save_pred_masks = pred_masks
            # ## opt2: save only the first frame of each temporal stride
            # save_frame_indices = key_frame_indices[::2]
            # save_pred_masks = pred_masks[::2]

            assert len(save_frame_indices) == save_pred_masks.shape[0]
            for i, frame_idx in enumerate(save_frame_indices):
                frame_filename = os.path.basename(image_file_list[frame_idx]).split(".")[0]
                save_path = "{}/{}.png".format(curr_mask_save_dir, frame_filename)
                cv2.imwrite(save_path, save_pred_masks[i])

        if predict_sam_mask:
            ## Step 3. Obtain query points and SAM masks
            ## use attention map to prompt SAM
            # video_heads_attn = video_heads_attn[:, ::2] # [b, t, h, w] ## TODO. stride 2 or not for full-frame mode
            points_list, labels_list, valid_nor_attn_weights_list, nor_attn_weights = get_candidate_points_video(video_heads_attn, sam_point_threshold)
            pred_masks = get_mask_preds_with_points_video_sam2(
                predictor,
                image_file_list,
                frame_file_path,
                original_size_list,
                key_frame_indices,
                points_list=points_list,
                labels_list=labels_list,
                nor_valid_attn_weights_list=valid_nor_attn_weights_list,
                nor_attn_weights=nor_attn_weights,
                final_score_threshold=sam_mask_score_thresh,
            )
            if pred_masks is None:
                print(f"No predictions for video {vid_id}, expression {exp_id}.")
                continue

            for i, pred_mask_vid in enumerate(pred_masks):
                assert len(image_file_list) == pred_mask_vid.shape[0]
                for frame_idx in range(len(image_file_list)):
                    frame_filename = os.path.basename(image_file_list[frame_idx]).split(".")[0]
                    save_path = "{}/{}.png".format(curr_mask_save_dir, frame_filename)
                    cv2.imwrite(save_path, pred_mask_vid[frame_idx])

        progress_bar.update(1)


if __name__ == "__main__":
    main(sys.argv[1:])
