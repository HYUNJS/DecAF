import sys

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from model.sam2_video_predictor_custom import build_sam2_video_predictor

from evaluation.inf_utils import (
    get_smooth_attn_map_thw,
    normalize_attn_map_thw,
    get_mask_preds_directly_from_attn,
    get_candidate_points_video,
    get_mask_preds_with_points_video_sam2
)
from evaluation.llm_prompting_utils import (
    obj_cls_predict_prompting,
    background_caption_prompting,
    obj_choice_prompting
)
import evaluation.llm_prompting_utils as llm_prompting_utils
from model.modeling_qwen2_5_vl_custom import Qwen2_5_VLForConditionalGenerationCustom

from demo_utils import *

transformers.logging.set_verbosity_error()  # hides transformers warnings


def main(args):
    args = parse_args(args)
    print(f"Arguments: {args}")

    # ------------------ initialize MLLM and SAM models -------------------------
    print("Initializing MLLM and SAM models...")

    num_input_frames = 16
    attn_layer_idx_start = 14
    attn_layer_idx_end = -1
    max_tokens_per_video, resize_scale = 400, 2
    llm_prompting_utils.MAX_BATCH = 8
    resized_height, resized_width = -1, -1
    dtype = torch.bfloat16

    model_class = Qwen2_5_VLForConditionalGenerationCustom
    model = model_class.from_pretrained(
        args.version,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
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

    predictor = build_sam2_video_predictor(args.vision_pretrained_config, args.vision_pretrained, device=torch.device("cuda"))
    sam_mask_score_thresh, sam_point_threshold = args.final_score_threshold, args.point_threshold

    # ------------------ preprocess video and text prompt -------------------------
    print("Preprocessing video and text prompt...")
    vis_save_path = args.vis_save_path
    if not os.path.exists(vis_save_path):
        os.makedirs(vis_save_path, exist_ok=True)
    video_path = args.video_path
    vid_name = args.video_path.split("/")[-1]
    frame_len = len(os.listdir(args.video_path))

    ## preprocess text prompt
    ref_query = args.exp.strip()
    if not ref_query.endswith("?") and not ref_query.endswith("."):
        ref_query += "."

    ## print video and exp info
    print(f"Video path: {video_path}, total frames: {frame_len}")
    print(f"Expression prompt: {ref_query}")

    ## frame uniform sampling
    if frame_len <= num_input_frames:
        key_frame_indices = list(range(frame_len))
    else:
        key_frame_indices = np.linspace(0, frame_len - 1, num_input_frames).astype(int).tolist()

    sampled_frames_cv, frame_file_path, original_size_list, all_frames_cv, image_file_list = load_and_process_video(video_path, key_frame_indices)

    # ------------------ attention map extraction via MLLM prompting -------------------------
    print("Extracting attention maps via MLLM prompting...")
    ## Step 1. LLM prompting to obtain attention maps
    size_per_tok = 28
    ## object class prediction prompting
    obj_cls_results_video = obj_cls_predict_prompting(model, processor, ref_query, frame_file_path, video_flag=True, tokens_per_frame=max_tokens_per_video)
    grid_t, grid_h, grid_w = obj_cls_results_video['visual_grid_thw'][0].tolist()
    grid_h, grid_w = grid_h // 2, grid_w // 2 # feature map size

    max_tokens_per_image = max_tokens_per_video * resize_scale * resize_scale 
    resized_height, resized_width = grid_h * size_per_tok * resize_scale, grid_w * size_per_tok * resize_scale

    obj_cls_results_frame = obj_cls_predict_prompting(model, processor, ref_query, frame_file_path, video_flag=False, tokens_per_frame=max_tokens_per_image, resized_height=resized_height, resized_width=resized_width)
    
    ## object class selection prompting
    obj_list = [*obj_cls_results_video['obj_cls_list'], *obj_cls_results_frame['obj_cls_list']]
    obj_name = obj_choice_prompting(model, processor, ref_query, obj_list)

    ## background captioning prompting
    bg_caption_results_video = background_caption_prompting(model, processor, obj_name, frame_file_path, video_flag=True, tokens_per_frame=max_tokens_per_video)
    grid_t, grid_h, grid_w = bg_caption_results_video['visual_grid_thw'][0].tolist()
    grid_h, grid_w = grid_h // 2, grid_w // 2 # feature map size

    max_tokens_per_image = max_tokens_per_video * resize_scale * resize_scale 
    resized_height, resized_width = grid_h * size_per_tok * resize_scale, grid_w * size_per_tok * resize_scale
    
    bg_caption_results_frame = background_caption_prompting(model, processor, obj_name, frame_file_path, video_flag=False, tokens_per_frame=max_tokens_per_image, resized_height=resized_height, resized_width=resized_width)

    llm_results = {
        "obj_cls_results_video": obj_cls_results_video,
        "obj_cls_results_frame": obj_cls_results_frame,
        "bg_caption_results_video": bg_caption_results_video,
        "bg_caption_results_frame": bg_caption_results_frame,
        "key_frame_indices": key_frame_indices,
    }

    ## Step 2. Fuse attention maps
    print("Fusing attention maps...")
    ## smooth the attention map
    attn_map_obj_video = get_smooth_attn_map_thw(llm_results, tgt_key="obj_cls_results_video", smooth_size=3)
    attn_map_obj_frame = get_smooth_attn_map_thw(llm_results, tgt_key="obj_cls_results_frame", smooth_size=7)
    attn_map_bg_video = get_smooth_attn_map_thw(llm_results, tgt_key="bg_caption_results_video", smooth_size=3)
    attn_map_bg_frame = get_smooth_attn_map_thw(llm_results, tgt_key="bg_caption_results_frame", smooth_size=7)

    ## object - background contrastive fusion
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

    ## video - frame complementary fusion
    _, img_h, img_w = norm_attn_map_frame.shape
    _norm_attn_map_video = F.interpolate(norm_attn_map_video.unsqueeze(1), size=(img_h, img_w), mode='bilinear').squeeze(1)            
    fused_attn_map = (_norm_attn_map_video + norm_attn_map_frame) / 2.0
    fused_attn_map = normalize_attn_map_thw(fused_attn_map, frame_wise_norm=False)

    assert len(key_frame_indices) == len(fused_attn_map), "Key frame indices length does not match attention map length."

    ## add batch-axis
    video_heads_attn = fused_attn_map.unsqueeze(0)
    video_heads_attn = video_heads_attn.cuda() if video_heads_attn.device == torch.device("cpu") else video_heads_attn

    ## Step 3. Obtain binary mask from attention weights
    print("Obtaining binary masks from attention weights...")
    assert video_heads_attn.shape[1] == len(key_frame_indices), (
        f"Attention map length {video_heads_attn.shape[1]} does not match key frame indices length {len(key_frame_indices)}."
    )
    binary_masks = get_mask_preds_directly_from_attn(video_heads_attn, original_size_list)

    ## Step 4. Obtain query points and SAM masks
    print("Obtaining query points and SAM masks...")
    ## use attention map to prompt SAM
    points_list, labels_list, valid_nor_attn_weights_list, nor_attn_weights = get_candidate_points_video(video_heads_attn, sam_point_threshold)
    sam_masks = get_mask_preds_with_points_video_sam2(
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
    )[0]

    # ------------------ visualization saving -------------------------
    print("Saving visualizations...")

    if args.save_per_frame:
        img_save_dir = os.path.join(vis_save_path, vid_name, "img")
        attn_save_dir = os.path.join(vis_save_path, vid_name, "attn_map")
        binary_mask_save_dir = os.path.join(vis_save_path, vid_name, "coarse_mask")
        point_save_dir = os.path.join(vis_save_path, vid_name, "points")
        sam_mask_save_dir = os.path.join(vis_save_path, vid_name, "sam_mask")

        ori_h, ori_w = original_size_list[0]
        new_h, new_w = ori_h // 2, ori_w // 2

        vis_img(sampled_frames_cv, (new_h, new_w), frame_file_path, img_save_dir)
        # vis_img(all_frames_cv, (new_h, new_w), image_file_list, img_save_dir)
        vis_attn_maps(fused_attn_map, sampled_frames_cv, (new_h, new_w), frame_file_path, attn_save_dir)
        vis_binary_mask(binary_masks, sampled_frames_cv, (new_h, new_w), frame_file_path, binary_mask_save_dir)
        vis_points(points_list, sampled_frames_cv, (new_h, new_w), frame_file_path, point_save_dir)
        # vis_sam_masks(sam_masks, sampled_frames_cv, (new_h, new_w), frame_file_path, sam_mask_save_dir, key_frame_indices)
        vis_sam_masks(sam_masks, all_frames_cv, (new_h, new_w), image_file_list, sam_mask_save_dir, None)

    else:
        all_save_path = os.path.join(vis_save_path, f"{vid_name}_all.png")
        vis_all_components(
            sampled_frames_cv,
            frame_file_path,
            key_frame_indices,
            original_size_list,
            fused_attn_map,
            points_list,
            binary_masks,
            sam_masks,
            ref_query,
            all_save_path
        )


if __name__ == "__main__":
    main(sys.argv[1:])