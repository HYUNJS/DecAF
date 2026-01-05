import argparse
import os
from glob import glob

import cv2
import numpy as np

from evaluation.inf_utils import *


def parse_args(args):
    parser = argparse.ArgumentParser(description="DecAF Demo")
    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument("--exp", required=True, type=str)
    parser.add_argument("--vis_save_path", default="./results/decaf_demo/", type=str)
    parser.add_argument("--save_per_frame", default=False, action="store_true")
    ## MLLM params
    parser.add_argument("--version", default="ckpts/Qwen2.5-VL-7B-Instruct", type=str)
    ## SAM2 params
    parser.add_argument(
        "--vision_pretrained", default="ckpts/sam2-hiera-large/sam2_hiera_large.pt", type=str
    )
    parser.add_argument("--vision_pretrained_config", default="sam2_hiera_l.yaml", type=str)
    parser.add_argument("--point_threshold", default=0.8, type=float)
    parser.add_argument("--final_score_threshold", default=0.8, type=float)

    return parser.parse_args(args)


def load_and_process_video(video_folder, key_frame_indices):
    if not os.path.exists(video_folder):
        print("File not found in {}".format(video_folder))
        raise FileNotFoundError

    image_file_list = sorted(glob(os.path.join(video_folder, f"*.jpg")))
    total_frames = len(image_file_list)
    frame_file_path, sampled_frames_cv = [], []
    image_ori_list = [
        cv2.cvtColor(cv2.imread(image_file_list[i]), cv2.COLOR_BGR2RGB) for i in range(total_frames)
    ]
    original_size_list = [image_ori.shape[:2] for image_ori in image_ori_list]

    for key_frm_idx in key_frame_indices:
        sampled_frames_cv.append(image_ori_list[key_frm_idx])
        frame_file_path.append(image_file_list[key_frm_idx])

    return (
        sampled_frames_cv,
        frame_file_path,
        original_size_list,
        image_ori_list,
        image_file_list,
    )


def vis_all_components(
    sampled_frames_cv,
    sampled_frames_file_path,
    key_frame_indices,
    original_size_list,
    final_fusion_attn_maps,
    points_list,
    binary_masks,
    sam_masks,
    exp,
    save_path
):
    ori_h, ori_w = original_size_list[0]
    new_h, new_w = ori_h // 4, ori_w // 4

    final_fusion_attn_maps = final_fusion_attn_maps.detach().cpu().numpy()   # (grid_t, grid_h, grid_w)
    final_fusion_attn_maps = (final_fusion_attn_maps * 255).astype(np.uint8)

    # binary_masks = binary_masks.cpu().numpy()
    # sam_masks = sam_masks.cpu().numpy()
    width_with_border = new_w * 5 + 40

    border_v = np.zeros((new_h, 10, 3), dtype=np.uint8)  # Vertical border
    border_h = np.zeros((10, width_with_border, 3), dtype=np.uint8)

    text_info = f"Exp: {exp}"
    text_img = np.zeros((100, width_with_border, 3), dtype=np.uint8) + 255
    cv2.putText(text_img, text_info, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    output_vis = text_img

    for i, key_idx in enumerate(key_frame_indices):
        img = sampled_frames_cv[i]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (new_w, new_h))

        r_final_fusion_attn_map = cv2.resize(final_fusion_attn_maps[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        overlay_final_fusion = cv2.addWeighted(img, 0.5, cv2.applyColorMap(r_final_fusion_attn_map, cv2.COLORMAP_JET), 0.5, 0)

        binary_mask = binary_masks[i]
        sam_mask = sam_masks[key_idx]

        binary_mask = cv2.resize(binary_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        sam_mask = cv2.resize(sam_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        overlay_binary_mask = cv2.addWeighted(img, 0.5, cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET), 0.5, 0)
        overlay_sam_mask = cv2.addWeighted(img, 0.5, cv2.applyColorMap(sam_mask, cv2.COLORMAP_JET), 0.5, 0)

        point_img = img.copy()
        pos_points = []
        rel_points = points_list[i]
        if rel_points is not None and len(rel_points) > 0:
            rel_points = rel_points.detach().cpu().numpy()
            for rel_point in rel_points:
                rel_x, rel_y = rel_point[0]
                rel_x = int(round(rel_x * new_w))
                rel_y = int(round(rel_y * new_h))
                pos_points.append((rel_y, rel_x))

        for pos_point in pos_points:
            y, x = pos_point
            cv2.circle(point_img, (x, y), 5, (0, 255, 0), -1)

        output_frame = np.concatenate(
            (
                img,
                border_v,
                overlay_final_fusion,
                border_v,
                overlay_binary_mask,
                border_v,
                point_img,
                border_v,
                overlay_sam_mask,
            ),
            axis=1
        )

        output_vis = np.concatenate((output_vis, border_h, output_frame), axis=0)

    # Save the final visualization
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, output_vis)


def vis_img(sampled_frames_cv, new_size, sampled_frames_file_path, save_dir):
    new_h, new_w = new_size
    os.makedirs(save_dir, exist_ok=True)

    for i, frame_cv in enumerate(sampled_frames_cv):
        filename = os.path.basename(sampled_frames_file_path[i]).split(".")[0]
        save_path = "{}/{}.png".format(save_dir, filename)

        img = frame_cv
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (new_w, new_h))

        cv2.imwrite(save_path, img)


def vis_attn_maps(attn_maps, sampled_frames_cv, new_size, sampled_frames_file_path, save_dir):
    new_h, new_w = new_size
    attn_maps = attn_maps.detach().cpu().numpy()
    attn_maps = (attn_maps * 255).astype(np.uint8)  # Convert to uint8 for visualization

    os.makedirs(save_dir, exist_ok=True)

    for i, frame_cv in enumerate(sampled_frames_cv):
        filename = os.path.basename(sampled_frames_file_path[i]).split(".")[0]
        save_path = "{}/{}.png".format(save_dir, filename)

        img = frame_cv
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (new_w, new_h))

        attn_map = attn_maps[i]
        attn_map = cv2.resize(attn_map, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        overlayed_img = cv2.addWeighted(img, 0.5, cv2.applyColorMap(attn_map, cv2.COLORMAP_JET), 0.5, 0)

        cv2.imwrite(save_path, overlayed_img)


def vis_sam_masks(sam_masks, sampled_frames_cv, new_size, sampled_frames_file_path, save_dir, key_frame_indices=None):
    new_h, new_w = new_size
    os.makedirs(save_dir, exist_ok=True)

    if key_frame_indices is None:
        key_frame_indices = range(len(sampled_frames_cv))

    for i, frame_idx in enumerate(key_frame_indices):
        frame_file = sampled_frames_file_path[i]
        filename = os.path.basename(frame_file).split(".")[0]
        save_path = "{}/{}.png".format(save_dir, filename)

        img = sampled_frames_cv[i]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (new_w, new_h))

        sam_mask = sam_masks[frame_idx]
        sam_mask = cv2.resize(sam_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        color_mask = np.zeros_like(img)
        color_mask[sam_mask > 0] = (0, 0, 255)

        # overlay the mask on the image
        alpha = 0.5
        overlayed_img = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)

        # Etract the contours of the mask
        contours, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the overlayed image
        cv2.drawContours(overlayed_img, contours, -1, (0, 0, 0), 1)

        cv2.imwrite(save_path, overlayed_img)


def vis_points(points_list, sampled_frames_cv, new_size, sampled_frames_file_path, save_dir):
    new_h, new_w = new_size
    os.makedirs(save_dir, exist_ok=True)

    for i, frame_cv in enumerate(sampled_frames_cv):
        filename = os.path.basename(sampled_frames_file_path[i]).split(".")[0]
        save_path = "{}/{}.png".format(save_dir, filename)

        img = frame_cv
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (new_w, new_h))

        pos_points = []
        rel_points = points_list[i]
        if rel_points is not None and len(rel_points) > 0:
            rel_points = rel_points.detach().cpu().numpy()
            for rel_point in rel_points:
                rel_x, rel_y = rel_point[0]
                rel_x = int(round(rel_x * new_w))
                rel_y = int(round(rel_y * new_h))
                pos_points.append((rel_y, rel_x))
        
        point_img = img.copy()
        for pos_point in pos_points:
            y, x = pos_point
            cv2.circle(point_img, (x, y), 5, (255, 0, 255), -1)  # Draw green circle for points

        cv2.imwrite(save_path, point_img)


def vis_binary_mask(binary_masks, sampled_frames_cv, new_size, sampled_frames_file_path, save_dir):
    new_h, new_w = new_size
    os.makedirs(save_dir, exist_ok=True)

    for i, frame_cv in enumerate(sampled_frames_cv):
        filename = os.path.basename(sampled_frames_file_path[i]).split(".")[0]
        save_path = "{}/{}.png".format(save_dir, filename)

        img = frame_cv
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (new_w, new_h))

        binary_mask = binary_masks[i]
        binary_mask = cv2.resize(binary_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        color_mask = np.zeros_like(img)
        color_mask[binary_mask > 0] = (0, 0, 255)

        # overlay the mask on the image
        alpha = 0.5
        overlayed_img = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)

        # Etract the contours of the mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the overlayed image
        cv2.drawContours(overlayed_img, contours, -1, (0, 0, 0), 1)

        cv2.imwrite(save_path, overlayed_img)


def save_text_file(exp, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"exp.txt")
    with open(save_path, "w") as f:
        f.write(f"Expression: {exp}\n")