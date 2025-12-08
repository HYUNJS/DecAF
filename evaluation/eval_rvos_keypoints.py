###########################################################################
# Created by: BUAA
# Email: clyanhh@gmail.com
# Copyright (c) 2024
###########################################################################
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import os.path as osp
import pickle
import time
import argparse
import cv2
import json
import traceback
import numpy as np
import multiprocessing as mp
import pandas as pd
from termcolor import colored
from tqdm import tqdm
from pycocotools import mask as cocomask
import torch
from evaluation.inf_utils import get_dataset

NUM_WORKERS = 16


def eval_queue(q, out_dict, pred_path):
    while not q.empty():
        vid_name, exp = q.get()
        vid = exp_dict[vid_name]
        exp_name = f"{vid_name}_{exp}"

        if len(os.listdir(osp.join(video_folder, vid_name))) != len(vid['frames']):
            print(f"[Frame-Mask Mismatch] {exp_name}")
            out_dict[exp_name] = [-1.0] * 5 # PP@1, 5, 10, 50, 100
            continue
            
        try:
            vid_path = osp.join(pred_path, vid_name)
            if not osp.exists(vid_path):
                print(f"{vid_name} not found")
                out_dict[exp_name] = [0.0] * 5 # PP@1, 5, 10, 50, 100
                continue
            
            pred_0_path = osp.join(pred_path, vid_name, exp, "keypoints.pt")
            pred_keypoints = torch.load(pred_0_path, weights_only=True)

            # Determine video H,W by loading any GT mask frame
            # (we need mask shape only)
            sample_mask_id = vid["expressions"][exp]["anno_id"][0]

            mask_rle = mask_dict[str(sample_mask_id)][0]
            sample_mask = cocomask.decode(mask_rle)
            if sample_mask.ndim == 3:   # (H,W,1)
                sample_mask = sample_mask[..., 0]
            h, w = sample_mask.shape


            # Load GT masks (combine all instances per frame)
            vid_len = len(vid["frames"])
            gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
            anno_ids = vid["expressions"][exp]["anno_id"]

            for frame_idx, frame_name in enumerate(vid["frames"]):
                # all instances in the same frame
                for anno_id in anno_ids:
                    mask_rle = mask_dict[str(anno_id)][frame_idx]
                    if mask_rle:
                        gt_masks[frame_idx] += cocomask.decode(mask_rle)
            
            # ----- Point Precision Measure ----- #
            Ks = [1, 5, 10, 50, 100]
            # offset = 25
            point_precisions = []

            # Sort points by score descending
            pred_keypoints = pred_keypoints[pred_keypoints[:, 3].argsort(descending=True)]

            for K in Ks:
                topk = pred_keypoints[:K]
                hits = 0

                for t, y, x, s in topk:
                    t = int(t)
                    py = min(int(y * h), h - 1)
                    px = min(int(x * w), w - 1)
                    min_py = max(py - offset, 0)
                    max_py = min(py + offset, h-1)
                    min_px = max(px - offset, 0)
                    max_px = min(px + offset, w-1)
                    
                    if t >= vid_len:
                        gt_masks
                    
                    # if gt_masks[t, py, px] > 0:
                    if (gt_masks[t, min_py:max_py, min_px:max_px]).any():
                        hits += 1

                point_precisions.append(hits / K)

            out_dict[exp_name] = point_precisions
            
        except:
            print(colored(f"error: {exp_name}, {traceback.format_exc()}", "red"))
            metrics = [0.0] * 5
            out_dict[exp_name] = metrics

def parse_keypoints_results(output_dict):
    data_list = []
    for videxp, (pp1, pp5, pp10, pp50, pp100) in output_dict.items():
        if pp1 == -1:
            continue

        vid_name, exp = videxp.rsplit("_", 1)
        data = {}

        data["video_name"] = vid_name
        data["exp_id"] = exp
        data["exp"] = exp_dict[vid_name]["expressions"][exp]["exp"]
        data["videxp"] = videxp
        data["PP@1"] = round(100 * pp1, 2)
        data["PP@5"] = round(100 * pp5, 2)
        data["PP@10"] = round(100 * pp10, 2)
        data["PP@50"] = round(100 * pp50, 2)
        data["PP@100"] = round(100 * pp100, 2)

        data_list.append(data)

    pp1 = np.array([d["PP@1"] for d in data_list]).mean()
    pp5 = np.array([d["PP@5"] for d in data_list]).mean()
    pp10 = np.array([d["PP@10"] for d in data_list]).mean()
    pp50 = np.array([d["PP@50"] for d in data_list]).mean()
    pp100 = np.array([d["PP@100"] for d in data_list]).mean()

    results = {
        "overall": {
            "PP@1": pp1,
            "PP@5": pp5,
            "PP@10": pp10,
            "PP@50": pp50,
            "PP@100": pp100,
        }
    }
    return results, data_list


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

if __name__ == "__main__":
    """
    Current version does not support ReVOS which needs to handle foregraound mask file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_path", type=str)
    parser.add_argument("--exp_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--foreground_mask_path", type=str, default="")
    parser.add_argument("--offset", type=int, default=25)
    
    args = parser.parse_args()
    is_davis = "davis" in args.exp_path.lower()
    offset = args.offset
    pred_path = args.pred_path
    pred_path = pred_path if pred_path[-1] != "/" else pred_path[:-1] # if end by separator, remove it
    dataset_name = "_".join(pred_path.split("/")[-1].split("_")[0:-1])
    video_folder, _, _ = get_dataset(dataset_name)

    queue = mp.Queue()
    output_dict = mp.Manager().dict()
    if is_davis:
        exp_dict = get_davis_meta(args.exp_path)
    else:
        exp_dict = json.load(open(args.exp_path))["videos"]
    if args.mask_path.endswith(".json"):
        mask_dict = json.load(open(args.mask_path))
    else:
        mask_dict = pickle.load(open(args.mask_path, "rb"))

    for vid_name in exp_dict:
        vid = exp_dict[vid_name]
        for exp in vid["expressions"]:
            queue.put([vid_name, exp])
            # queue.append([vid_name, exp])

    start_time = time.time()
    if NUM_WORKERS > 1:
        processes = []
        for rank in range(NUM_WORKERS):
            p = mp.Process(target=eval_queue, args=(queue, output_dict, pred_path))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        eval_queue(queue, output_dict, pred_path)

    results, data_list = parse_keypoints_results(output_dict)
    
    ## save average results
    # eval_dataset, eval_model_cfg = pred_path.split("/")[-2:]
    # SAVE_ROOT = osp.join("metrics", eval_model_cfg)
    # os.makedirs(SAVE_ROOT, exist_ok=True)
    # output_json_path = osp.join(SAVE_ROOT, f"{eval_dataset}.json")
    # output_csv_path = osp.join(SAVE_ROOT, f"{eval_dataset}.csv")
    save_path = pred_path.replace("outputs", "metrics")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output_json_path = save_path + f"_offset-{offset}.json"
    output_csv_path = save_path + f"_offset-{offset}.csv"

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_json_path}")

    data4csv = {}
    for data in data_list:
        for k, v in data.items():
            data4csv[k] = data4csv.get(k, []) + [v]

    df = pd.DataFrame(data4csv)
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" % (total_time))



    # # ----- verify the same number of frames and masks ----- #
    # frame_rootdir = "datasets/RVOSJoint/ReasonVOS/JPEGImages/"
    # mask_rootdir = "datasets/RVOSJoint/ReasonVOS/Annotations/"

    # vid_and_anno_id_list = []
    # for vid_name in exp_dict:
    #     vid = exp_dict[vid_name]
    #     for exp in vid["expressions"]:
    #         vid_and_anno_id_list.append([vid_name, vid["expressions"][exp]['anno_id'][0]])

    # for frame_dirname, mask_dirname in vid_and_anno_id_list:
    #     num_frames = len(os.listdir(osp.join(frame_rootdir, frame_dirname)))
    #     num_masks = len(os.listdir(osp.join(mask_rootdir, mask_dirname)))

    #     if num_frames != num_masks:
    #         print(f"Mismatch frames and masks: {frame_dirname}, {num_frames}, {num_masks}")
