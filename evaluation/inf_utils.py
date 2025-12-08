import os
import argparse
import json
import pickle
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm, trange
import torch
import einops

from scipy.ndimage import (
    binary_closing,
    generate_binary_structure,
    center_of_mass,
    label,
    gaussian_laplace,
)
from collections import defaultdict

# import torch.functional
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from pycocotools import mask as cocomask

from evaluation.sam2_utils import *


EPS = 1e-8

YTVOS_DATA_ROOT = "datasets/RVOSJoint/ref-youtube-vos"
DAVIS_DATA_ROOT = "datasets/RVOSJoint/davis17"
MEVIS_DATA_ROOT = "datasets/RVOSJoint/mevis"
REVOS_DATA_ROOT = "datasets/RVOSJoint/ReVOS"
REASONVOS_DATA_ROOT = "datasets/RVOSJoint/ReasonVOS"


PROMPT_TMPL_OBJ_CLS = (
    "{sent} \n"
    "What is the main object (or objects) referred to in the given expression or question? \n"
    "Focus on the **primary subject or agent** involved in the described action or behavior. "
    "Respond with a single word (e.g., 'cat', 'person', 'dog') that best describes the target object(s). "
)

PROMPT_TMPL_OBJ_CLS_abl_v1 = (
    "{sent} \n"
    "Identify the primary object referred to in the expression and answer with a single word. \n"
)

PROMPT_TMPL_OBJ_CLS_abl_v2 = (
    "{sent} \n"
    "Identify the primary object referred to in the expression. \n"
    "Focus on the **primary subject or agent** involved in the described action or behavior. "
    "Respond with a single word (e.g., 'cat', 'person', 'dog') that best describes the target object(s). "
)

PROMPT_TMPL_OBJ_CLS_abl_v3 = (
    "{sent} \n"
    "Determine the primary subject or agent mentioned in the expression or question, and provide the object's label within a single word or phrase. "
)

PROMPT_TMPL_BACKGROUND = (
    "Describe the background scene of the video, excluding any {obj_name}. Answer the question using a single word or phrase."
)

PROMPT_TMPL_BACKGROUND_abl_v1 = (
    "Describe the background scene of the video. Answer the question using a single word or phrase."
)

PROMPT_TMPL_BACKGROUND_abl_v2 = (
    "Describe the background of the video while excluding any {obj_name}, using a single word or short phrase."
)

PROMPT_TMPL_BACKGROUND_abl_v3 = (
    "Describe the background scene of the video, excluding the objects referred to in the given expression or question '{sent}'. Answer the question using a single word or phrase."
)

PROMPT_TMPL_OBJ_CHOICE = (
    "Given:\n"
    "- Expression: '{ref_query}'\n"
    "- Candidate object class list: {obj_cls_list}\n"
    "Goal: Identify the object class referred to by the expression. Instructions:\n"
    "1. If the expression is **clear**, rely on it directly (e.g., 'a person driving a car' → 'person').\n"
    "2. If the expression is **vague**, use the object class list to support your decision (e.g., check frequency and plausibility).\n"
    "3. Avoid defaulting to the most frequent class unless the expression lacks clarity.\n"
    "Output the most likely referred object class - just the label.\n"
)

PROMPT_TMPL_OBJ_CHOICE_abl_v1 = (
    "Given the expression ‘{ref_query}’ and the candidate object classes {obj_cls_list}, select the single class label that best matches the object referred to in the expression.\n"
)

PROMPT_TMPL_OBJ_CHOICE_abl_v2 = (
    "Using the expression ‘{ref_query}’ and the candidate object classes {obj_cls_list}, determine which object class the expression refers to.\n"
    "If the reference is explicit, rely on the expression; if ambiguous, use the class list as support.\n"
    "Output only the most likely object class.\n"
)

PROMPT_TMPL_OBJ_CHOICE_abl_v3 = (
    "Given:\n"
    "- Expression: '{ref_query}'\n"
    "Goal: Identify the object class referred to by the expression. (e.g., 'a person driving a car' → 'person').\n"
    "Output the most likely referred object class - just the label.\n"
)

SYS_QWEN2 = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a helpful assistant."
            }
        ]
    }
]

SYS_INTERNVL = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
            }
        ]
    }
]

def get_dataset(tgt_dataset, testset=False):

    # assert tgt_dataset == "revos" or tgt_dataset == "davis"

    if tgt_dataset == "davis":
        meta_exp_path = os.path.join(
            DAVIS_DATA_ROOT,
            "meta_expressions",
            "valid",
            "meta_expressions.json",
        )
        video_folder = os.path.join(DAVIS_DATA_ROOT, "valid/JPEGImages")
        mask_path = os.path.join(DAVIS_DATA_ROOT, "valid/mask_dict.pkl")
        mask_dict = pickle.load(open(mask_path, "rb"))
    elif tgt_dataset == "revos":
        meta_exp_path = os.path.join(
            REVOS_DATA_ROOT,
            "meta_expressions_valid_.json"
        )
        video_folder = os.path.join(REVOS_DATA_ROOT, "JPEGImages")
        mask_path = os.path.join(REVOS_DATA_ROOT, "mask_dict.json")
        mask_dict = json.load(open(mask_path, "r"))
    elif tgt_dataset == "ytvos":
        meta_exp_path = os.path.join(
            YTVOS_DATA_ROOT, "meta_expressions/valid/meta_expressions.json"
        )
        video_folder = os.path.join(YTVOS_DATA_ROOT, "valid/JPEGImages")
        mask_dict = None
    elif tgt_dataset == "mevis":
        subset_type = "valid" if testset else "valid_u"
        meta_exp_path = os.path.join(MEVIS_DATA_ROOT, subset_type, "meta_expressions.json")
        video_folder = os.path.join(MEVIS_DATA_ROOT, subset_type, "JPEGImages")
        mask_path = os.path.join(MEVIS_DATA_ROOT, subset_type, "mask_dict.json")
        mask_dict = None if testset else json.load(open(mask_path, "r"))
    elif tgt_dataset == "reasonvos":
        meta_exp_path = os.path.join(REASONVOS_DATA_ROOT, "meta_expressions_v2.json")
        video_folder = os.path.join(REASONVOS_DATA_ROOT, "JPEGImages")
        mask_dict = None
    else:
        raise NotImplementedError(f"Dataset {tgt_dataset} is not implemented.")

    if tgt_dataset == "davis":
        meta_exp = get_davis_meta(meta_exp_path)
    else:
        meta_exp = json.load(open(meta_exp_path, "r"))["videos"]

    return video_folder, meta_exp, mask_dict

def parse_args(args):
    parser = argparse.ArgumentParser(description="DecAF Inference")
    parser.add_argument(
        "--dataset", required=True, type=str, choices=["davis", "ytvos", "mevis", "revos", "reasonvos"]
    )
    parser.add_argument("--testset", action="store_true", default=False)  # for mevis
    parser.add_argument("--version", required=True, type=str, help="PATH/TO/MODEL")
    parser.add_argument("--vis_save_path", default="./results/decaf_qwen/", type=str)
    parser.add_argument("--save_overlay", action="store_true", default=False)
    parser.add_argument("--subset_num", default=1, type=int)
    parser.add_argument("--subset_idx", default=0, type=int)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )

    ## SAM2 params
    parser.add_argument(
        "--vision_pretrained", default="ckpts/sam2-hiera-large/sam2_hiera_large.pt", type=str
    )
    parser.add_argument("--vision_pretrained_config", default="sam2_hiera_l.yaml", type=str)
    
    ## Optional params (in dev)
    parser.add_argument("--num_input_frames", default=16, type=int)
    parser.add_argument("--attn_layer_idx_start", default=14, type=int, help="starting layer index for attn rollout")
    parser.add_argument("--attn_layer_idx_end", default=-1, type=int, help="ending layer index for attn rollout")

    parser.add_argument("--predict_attn_mask", action="store_true", default=False)
    parser.add_argument("--predict_sam_mask", action="store_true", default=False)
    parser.add_argument("--attn_save_path", default=None, type=str, help="path to save attention maps")
    parser.add_argument("--use_saved_attn", action="store_true", default=False)
    parser.add_argument("--attn_resize_scale", default=1, type=int)
    parser.add_argument("--attn_max_token", default=768, type=int)
    parser.add_argument("--attn_max_token_per_image", default=-1, type=int) # 16384
    parser.add_argument("--attn_max_batch", default=16, type=int)
    parser.add_argument("--max_patches", default=12, type=int)
    parser.add_argument("--prompt_obj_ver", default=0, type=int)
    parser.add_argument("--prompt_bg_ver", default=0, type=int)
    parser.add_argument("--prompt_obj_cnt_ver", default=0, type=int)
    
    parser.add_argument("--point_threshold", default=0.8, type=float)
    parser.add_argument("--final_score_threshold", default=0.8, type=float)

    return parser.parse_args(args)


def load_and_process_video(video_folder, vid_id, dtype, key_frame_indices):
    image_folder = os.path.join(video_folder, vid_id)
    if not os.path.exists(image_folder):
        print("File not found in {}".format(image_folder))
        raise FileNotFoundError

    image_file_list = sorted(glob(os.path.join(image_folder, f"*.jpg")))
    total_frames = len(image_file_list)
    frame_file_path, image_list_ori = [], []
    image_ori_list = [
        cv2.cvtColor(cv2.imread(image_file_list[i]), cv2.COLOR_BGR2RGB) for i in range(total_frames)
    ]
    original_size_list = [image_ori.shape[:2] for image_ori in image_ori_list]

    for key_frm_idx in key_frame_indices:
        image_list_ori.append(image_ori_list[key_frm_idx])
        frame_file_path.append(image_file_list[key_frm_idx])

    return (
        image_list_ori,
        frame_file_path,
        original_size_list,
        image_file_list,
    )


def get_sharegpt_format_messages(question, video_frame_path_list, is_video=False, tokens_per_frame=768, resized_height=-1, resized_width=-1):
    if video_frame_path_list is None:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            },
        ]
        return messages

    if is_video:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_frame_path_list,
                        "max_pixels": tokens_per_frame * 28 * 28,
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": video_frame_path_list[0],
                        "max_pixels": tokens_per_frame * 28 * 28,
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
        ]

    if resized_height > 0 and resized_width > 0:
        messages[0]['content'][0]['resized_height'] = resized_height
        messages[0]['content'][0]['resized_width'] = resized_width
    
    return messages

def get_internvl_sharegpt_format_messages(question, video_frame_path_list, is_video=False):
    if video_frame_path_list is None:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            },
        ]
        return messages

    if is_video:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_frame_path_list,
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": video_frame_path_list[0],
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
        ]
    
    return messages

def get_sharegpt_format_text_messages(question):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question,
                },
            ],
        },
    ]

    return messages


def get_gt_masks(mask_dict, anno_ids, original_size_list, frame_len):
    if len(anno_ids) > 0:
        frame_len = len(mask_dict[str(anno_ids[0])])
        h, w = None, None
        for i in range(frame_len):
            mask_rle = mask_dict[str(anno_ids[0])][i]
            if mask_rle:
                h, w = mask_rle["size"]
                break
    else:
        h, w = original_size_list[0]

    new_h, new_w = h // 2, w // 2

    gt_masks = np.zeros((frame_len, h, w), dtype=np.uint8)

    num_object = 0

    for i in range(frame_len):
        temp_num_object = 0
        for anno_id in anno_ids:
            mask_rle = mask_dict[str(anno_id)][i]
            if mask_rle:
                gt_masks[i] += cocomask.decode(mask_rle)
                temp_num_object += 1
        if temp_num_object > num_object:
            num_object = temp_num_object

    gt_masks_np = (gt_masks > 0).astype(np.uint8)

    return gt_masks_np, (new_h, new_w)


def get_davis_meta(mevis_exp_path):
    # read expression data
    with open(str(mevis_exp_path), "r") as f:
        subset_expressions_by_video = json.load(f)["videos"]
    videos = sorted(list(subset_expressions_by_video.keys()))

    metas = []
    anno_count = 0  # serve as anno_id
    for vid in videos:
        vid_data = subset_expressions_by_video[vid]
        vid_frames = sorted(vid_data["frames"])
        vid_len = len(vid_frames)

        exp_id_list = sorted(list(vid_data["expressions"].keys()))
        for exp_id in exp_id_list:
            subset_expressions_by_video[vid]["expressions"][exp_id]["anno_id"] = [
                anno_count,
            ]
            anno_count += 1

    return subset_expressions_by_video


def get_smoothed_attn_map(attn_map: torch.Tensor, kernel_size=7, sigma=1.0) -> torch.Tensor:
    """
    attn_map: shape (T, H, W)
    returns smoothed map, same shape
    """
    if attn_map.dim() == 2:
        attn_map = attn_map.unsqueeze(0)  # (1, H, W)
    smoothed = apply_gaussian_smoothing(attn_map, kernel_size, sigma)  # (T, H, W)
    return smoothed

def get_smooth_attn_map_thw(llm_results, tgt_key, smooth_size):
    attn_weights_list = llm_results[tgt_key]["attn_weights_list"]
    visual_token_mask = llm_results[tgt_key]["visual_token_mask"]
    grid_thw = llm_results[tgt_key]["visual_grid_thw"]

    is_video = True if "_video" in tgt_key else False
    attn_weights = attn_weights_list[-1].float()
    grid_t, grid_h, grid_w = grid_thw[0].tolist()
    grid_t = grid_t if is_video else len(attn_weights)
    grid_h, grid_w = grid_h // 2, grid_w // 2
    grid_thw = (grid_t, grid_h, grid_w)
    attn_weights_thw = einops.rearrange(attn_weights[visual_token_mask], "(t h w) -> t h w", t=grid_t, h=grid_h, w=grid_w).to("cuda")
    smooth_attn_weights_thw = get_smoothed_attn_map(attn_weights_thw, kernel_size=smooth_size, sigma=1.0)
    
    return smooth_attn_weights_thw

def get_smooth_attn_map_thw_llava(llm_results, tgt_key, smooth_size):
    attn_weights_list = llm_results[tgt_key]["attn_weights_list"]
    visual_token_mask = llm_results[tgt_key]["visual_token_mask"]
    grid_thw = llm_results[tgt_key]["visual_grid_thw"]

    is_video = True if "_video" in tgt_key else False
    attn_weights = attn_weights_list[-1].float()
    grid_t, grid_h, grid_w = grid_thw[0].tolist()
    grid_t = grid_t if is_video else len(attn_weights)
    grid_thw = (grid_t, grid_h, grid_w)
    attn_weights_thw = einops.rearrange(attn_weights[visual_token_mask], "(t h w) -> t h w", t=grid_t, h=grid_h, w=grid_w).to("cuda")
    smooth_attn_weights_thw = get_smoothed_attn_map(attn_weights_thw, kernel_size=smooth_size, sigma=1.0)
    
    return smooth_attn_weights_thw

def get_smooth_attn_map_thw_internvl(llm_results, tgt_key, smooth_size):
    attn_weights_list = llm_results[tgt_key]["attn_weights_list"]
    visual_token_mask = llm_results[tgt_key]["visual_token_mask"]
    grid_thw = llm_results[tgt_key]["visual_grid_thw"]
    if "tiling_hw" in llm_results[tgt_key]:
        tiling_hw = llm_results[tgt_key]["tiling_hw"]
    else:
        tiling_hw = None

    is_video = True if "_video" in tgt_key else False
    attn_weights = attn_weights_list[-1].float()
    grid_t, grid_h, grid_w = grid_thw[0].tolist()
    tiling_h, tiling_w = tiling_hw[0].tolist() if tiling_hw is not None else (1, 1)
    grid_t = grid_t if is_video else len(attn_weights)
    # grid_h, grid_w = grid_h // 2, grid_w // 2

    if is_video or (tiling_h == 1 and tiling_w == 1):
        attn_weights_thw = einops.rearrange(attn_weights[visual_token_mask], "(t h w) -> t h w", t=grid_t, h=grid_h, w=grid_w).to("cuda")
    else:
        visual_token_num_per_b = visual_token_mask.sum(dim=-1)[0].item()
        attn_weights_t = attn_weights[visual_token_mask].reshape(grid_t, visual_token_num_per_b).to("cuda")
        num_tiling = tiling_h * tiling_w
        num_token_num_per_tile = grid_h * grid_w
        assert visual_token_num_per_b == num_tiling * num_token_num_per_tile, f"visual_token_num_per_b {visual_token_num_per_b} != num_tiling {num_tiling} * num_token_num_per_tile {num_token_num_per_tile}"
        attn_weights_t_tiles = attn_weights_t.reshape(grid_t, num_tiling, num_token_num_per_tile)
        attn_weights_thw = einops.rearrange(attn_weights_t_tiles, "t (th tw) (gh gw) -> t (th gh) (tw gw)", th=tiling_h, tw=tiling_w, gh=grid_h, gw=grid_w)

    smooth_attn_weights_thw = get_smoothed_attn_map(attn_weights_thw, kernel_size=smooth_size, sigma=1.0)
    
    return smooth_attn_weights_thw

def normalize_attn_map_thw(attn_map, frame_wise_norm):
    if frame_wise_norm:
        t, h, w = attn_map.shape
        attn_map_hw = attn_map.flatten(1)
        min_v = attn_map_hw.min(dim=-1, keepdim=True)[0] # [t, 1]
        max_v = attn_map_hw.max(dim=-1, keepdim=True)[0] # [t, 1]
        min_max_diff = max_v - min_v
        # Use the true range when > 0; otherwise use 1 so the normalized result becomes 0 for constant frames
        safe_denominator = torch.where(min_max_diff > 0, min_max_diff, torch.ones_like(min_max_diff))
        attn_map_hw_norm = (attn_map_hw - min_v) / safe_denominator
        # attn_map_hw_norm = (attn_map_hw - min_v) / (max_v - min_v + EPS)
        attn_map_norm = attn_map_hw_norm.reshape(t, h, w)
    else:
        min_v = attn_map.min()
        max_v = attn_map.max()
        min_max_diff = max_v - min_v
        safe_denominator = torch.where(min_max_diff > 0, min_max_diff, torch.ones_like(min_max_diff))
        # attn_map_norm = (attn_map - min_v) / (max_v - min_v + EPS)
        attn_map_norm = (attn_map - min_v) / safe_denominator
    
    return attn_map_norm


def get_mask_preds_directly_from_attn(video_attn_weights_layers, original_size_list):
    """
    Get mask predictions directly from attention weights.
    """
    ori_h, ori_w = original_size_list[0]
    video_attn_weights = video_attn_weights_layers[-1]  # (grid_t, grid_h, grid_w)
    video_attn_weights_resized = F.interpolate(video_attn_weights.unsqueeze(1), size=(ori_h, ori_w), mode='bilinear', align_corners=False).squeeze(1)
    video_attn_weights_resized = (video_attn_weights_resized * 255).cpu().numpy().astype(np.uint8)
    grid_t = video_attn_weights_resized.shape[0]
    
    pred_masks = np.zeros((grid_t, ori_h, ori_w), dtype=np.uint8)
    
    for t in range(grid_t):
        attn_map = video_attn_weights_resized[t]
        _, binary_mask = cv2.threshold(attn_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pred_masks[t] = binary_mask
    
    return pred_masks


def get_mask_preds_with_points_video_sam2(
    predictor,
    full_frame_file_paths,
    frame_file_paths,
    original_size_list,
    key_frame_indices,
    points_list,
    labels_list,
    nor_valid_attn_weights_list,
    nor_attn_weights,
    final_score_threshold=0.8,
):
    """
    Get mask predictions through sam2 official demo code.
    """
    num_frames_sam = len(full_frame_file_paths)
    pred_masks = []

    ori_h, ori_w = original_size_list[0]
    pred_masks_cur_vid = torch.zeros((num_frames_sam, ori_h, ori_w), device=nor_attn_weights.device)
    pred_masks_len = 0
    mask_prompt = []
    mask_cond_frame_indices = []
    mask_tracklet_dict_list = []
    # mask_tracklet_dict["mask_tracklet"]: (T, H, W)
    # mask_tracklet_dict["score"]: (1, )
    # mask_tracklet_dict["frame_idx"]: (1, )

    if points_list is None:
        # No points provided, return empty masks
        empty_mask = np.zeros((num_frames_sam, ori_h, ori_w)).astype(np.uint8)
        return [empty_mask]
    
    inference_state = predictor.init_state(frame_file_paths)
    for t in range(len(key_frame_indices)):
        predictor.reset_state(inference_state)
        points = points_list[t]  # (num_obj, 1, 2)
        labels = labels_list[t]  # (num_obj, 1)
        nor_valid_attn_weights = nor_valid_attn_weights_list[t]  # (num_obj, )
        grid_t, grid_h, grid_w = nor_attn_weights.shape

        if points is None:
            continue

        assert points.shape[0] == labels.shape[0] == nor_valid_attn_weights.shape[0]

        num_obj = points.shape[0]

        if num_obj == 0:
            raise NotImplementedError(
                "No objects found in the key frame. Please check the input points and labels."
            )
        
        pred_masks_from_points, obj_scores, delete_mask_tracklet = get_mask_tracklet_from_point(
            predictor,
            inference_state,
            points_prompt=points,
            labels_prompt=labels,
            frame_idx=t,
            attn_score=nor_valid_attn_weights,
            ori_size=(ori_h, ori_w),
            mask_tracklet_dict_list=mask_tracklet_dict_list,
        )

        if pred_masks_from_points is None:
            # No valid masks found for this frame, skip to the next frame
            continue
        # pred_masks_from_points: (keep_num_obj, T, ori_h, ori_w)
        keep_num_obj = pred_masks_from_points.shape[0]
        pred_masks_from_points = (pred_masks_from_points.sigmoid() > 0.5).float()
        # pred_masks_from_points = (
        #     F.interpolate(
        #         pred_masks_from_points,
        #         size=(ori_h, ori_w),
        #         mode="bilinear",
        #         align_corners=False,
        #     ).sigmoid()
        #     > 0.5
        # ).float()
        
        assert pred_masks_from_points.shape[1] == len(key_frame_indices)

        if len(delete_mask_tracklet) > 0:
            mask_tracklet_dict_list = [mask_tracklet_dict for i, mask_tracklet_dict in enumerate(mask_tracklet_dict_list) if i not in delete_mask_tracklet]

        for i in range(keep_num_obj):
            mask_tracklet = pred_masks_from_points[i]
            score = obj_scores[i]
            mask_tracklet_dict = {
                "mask_tracklet": mask_tracklet,
                "score": score,
                "frame_idx": t,
            }
            mask_tracklet_dict_list.append(mask_tracklet_dict)

    ac_scores = get_AC_score(nor_attn_weights, mask_tracklet_dict_list)
    mask_prompt, mask_cond_frame_indices = get_final_obj_prompts(
        mask_tracklet_dict_list,
        ac_scores,
        key_frame_indices,
        final_score_threshold,
    )

    grouped_masks = defaultdict(list)

    for f_idx, mask in zip(mask_cond_frame_indices, mask_prompt):
        grouped_masks[f_idx].append(mask)

    inference_states = predictor.init_state(full_frame_file_paths)
    for f_idx, masks in grouped_masks.items():
        predictor.reset_state(inference_states)
        masks = torch.cat(masks, dim=0)  # (N, H, W)
        # (obj_num, T, h', w')
        pred_masks_per_obj = get_video_mask_from_mask(predictor, inference_states, masks, f_idx)
        # pred_masks_per_obj: (obj_num, num_frames_sam, ori_h, ori_w)
        pred_masks_cur_vid += (
            (pred_masks_per_obj.sigmoid() > 0.5)
            .float().sum(dim=0)
        )
        # pred_masks_cur_vid += (
        #     (F.interpolate(pred_masks_per_obj, size=(ori_h, ori_w), mode="bilinear", align_corners=False).sigmoid() > 0.5)
        #     .float().sum(dim=0)
        # )
    torch.cuda.empty_cache()
    pred_masks.append(pred_masks_cur_vid)

    pred_masks_np = []
    for i, pred_mask_vid in enumerate(pred_masks):
        pred_mask_vid_np = pred_mask_vid.cpu().numpy()
        assert num_frames_sam == pred_mask_vid_np.shape[0]
        pred_mask_vid_np = (np.where(pred_mask_vid_np > 0, 1, 0) * 255).astype(np.uint8)
        pred_masks_np.append(pred_mask_vid_np)

    return pred_masks_np


def get_rel_bboxes(bboxes, sr_w, sr_h):
    rel_bboxes = []
    for bbox in bboxes:
        sr_x1, sr_y1, sr_x2, sr_y2 = bbox

        if sr_x1 >= sr_x2 or sr_y1 >= sr_y2:
            continue
        sr_x1 = max(0, min(sr_x1, sr_w - 1))
        sr_y1 = max(0, min(sr_y1, sr_h - 1))
        sr_x2 = max(1, min(sr_x2, sr_w))
        sr_y2 = max(1, min(sr_y2, sr_h))

        rel_sr_x1 = sr_x1 / sr_w
        rel_sr_y1 = sr_y1 / sr_h
        rel_sr_x2 = sr_x2 / sr_w
        rel_sr_y2 = sr_y2 / sr_h
        rel_bboxes.append([rel_sr_x1, rel_sr_y1, rel_sr_x2, rel_sr_y2])
    return rel_bboxes


def split_eval_data(video_folder, meta_exp, subset_num, subset_idx, vis_save_path=None, attn_save_path=None):
    job_list = []
    # vid_id_list = os.listdir(video_folder)
    vid_id_list = list(meta_exp.keys())
    for vid_id in vid_id_list:
        video_path = os.path.join(video_folder, vid_id)
        if not os.path.exists(video_path):
            print(f"[Not Found] {video_path}")
            continue
        for exp_id in list(meta_exp[vid_id]["expressions"].keys()):
            job_list.append((vid_id, exp_id))

    ## detect already predicted ones
    if vis_save_path is not None:
        new_job_list = []
        for vid_id, exp_id in job_list:
            curr_mask_save_dir = os.path.join(vis_save_path, vid_id, exp_id)
            if not os.path.exists(curr_mask_save_dir):
                new_job_list.append((vid_id, exp_id))
                continue
            mask_len = len(os.listdir(curr_mask_save_dir))
            frame_len = len(os.listdir(os.path.join(video_folder, vid_id)))
            vid_name = meta_exp[vid_id]["vid_id"] if "vid_id" in meta_exp[vid_id] else vid_id
            sam_mask_done_flag = mask_len == frame_len
            if not sam_mask_done_flag:
                new_job_list.append((vid_id, exp_id))
            else:
                print(f"Skip {vid_name}_{exp_id}")
        job_list = new_job_list
    
    if attn_save_path is not None:
        new_job_list = []
        for vid_id, exp_id in job_list:
            vid_name = meta_exp[vid_id]["vid_id"] if "vid_id" in meta_exp[vid_id] else vid_id
            curr_data_filepath = os.path.join(attn_save_path, f"{vid_name}_{exp_id}.pt")
            if not os.path.exists(curr_data_filepath):
                new_job_list.append((vid_id, exp_id))
                continue
            else:    
                print(f"Skip {vid_name}_{exp_id}")
        job_list = new_job_list
    
    job_list_subset = [job_list[i] for i in range(len(job_list)) if i % subset_num == subset_idx]

    return job_list_subset


def get_dtype(dtype_name):
    if dtype_name == "fp32":
        return torch.float
    elif dtype_name == "bf16":
        return torch.bfloat16
    elif dtype_name == "fp16":
        return torch.float16
    else:
        raise NotImplementedError(f"{dtype_name} is not defined.")


def gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """2D Gaussian kernel generation."""
    ax = torch.arange(kernel_size) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def apply_gaussian_smoothing(att_map: torch.Tensor, kernel_size=7, sigma=1.0) -> torch.Tensor:
    """
    att_map: shape (T, H, W)
    returns smoothed map, same shape
    """
    # if att_map.dim() == 2:
    #     att_map = att_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    att_map = att_map.unsqueeze(1)  # (T, 1, H, W)

    kernel = gaussian_kernel(kernel_size, sigma).to(att_map.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)

    padding = kernel_size // 2
    smoothed = F.conv2d(att_map, kernel, padding=padding)  # (T, 1, H, W)
    return smoothed.squeeze(1)  # shape (T, H, W)


def get_candidate_points_video(smoothed_layers_list, point_threshold=0.7):
    # smoothed_layers_list: [num_layers, grid_t, gird_h, grid_w]
    grid_t, grid_h, grid_w = smoothed_layers_list.shape[1:]
    nor_smoothed_list = smoothed_layers_list[-1]  # [grid_t, grid_h, grid_w]

    points_list = []
    labels_list = []
    valid_nor_attn_weights_list = []
    thw_indices = torch.nonzero(nor_smoothed_list >= point_threshold, as_tuple=False)  # (N, 3)

    for t in range(grid_t):
        indices = thw_indices[thw_indices[:, 0] == t][:, 1:]
        if indices.shape[0] == 0:
            points_list.append(None)
            labels_list.append(None)
            valid_nor_attn_weights_list.append(None)
            continue
        ys = (indices[:, 0].float() + 0.5) / grid_h  # i = row
        xs = (indices[:, 1].float() + 0.5) / grid_w  # j = col
        points = torch.stack([xs, ys], dim=1)  # (N, 2)
        points = points.unsqueeze(1).float()  # (N, 1, 2)
        points_list.append(points)
        labels = torch.ones(points.shape[0], dtype=torch.int32, device=points.device)  # (N,)
        labels = labels.unsqueeze(1)  # (N, 1)
        labels_list.append(labels)
        valid_nor_attn_weights = nor_smoothed_list[t][indices[:, 0], indices[:, 1]]  # (N,)
        valid_nor_attn_weights_list.append(valid_nor_attn_weights)

    return points_list, labels_list, valid_nor_attn_weights_list, nor_smoothed_list
