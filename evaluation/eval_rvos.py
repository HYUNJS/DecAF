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
from metrics import db_eval_iou, db_eval_boundary, get_r2vos_accuracy, get_r2vos_robustness

NUM_WORKERS = 16


def eval_queue(q, out_dict, pred_path):
    while not q.empty():
        vid_name, exp = q.get()
        vid = exp_dict[vid_name]
        exp_name = f"{vid_name}_{exp}"

        try:
            vid_path = osp.join(pred_path, vid_name)
            if not osp.exists(vid_path):
                print(f"{vid_name} not found")
                if mask_dict_foreground:
                    out_dict[exp_name] = [0.0, 0.0, 0.0, 0.0]
                else:
                    out_dict[exp_name] = [0.0, 0.0]
                continue

            pred_0_path = osp.join(pred_path, vid_name, exp, f"{vid['frames'][0]}.png")
            pred_0 = cv2.imread(pred_0_path, cv2.IMREAD_GRAYSCALE)
            h, w = pred_0.shape
            vid_len = len(vid["frames"])
            gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
            pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
            if mask_dict_foreground:
                foreground_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

            anno_ids = vid["expressions"][exp]["anno_id"]

            for frame_idx, frame_name in enumerate(vid["frames"]):
                # all instances in the same frame
                for anno_id in anno_ids:
                    mask_rle = mask_dict[str(anno_id)][frame_idx]
                    if mask_rle:
                        gt_masks[frame_idx] += cocomask.decode(mask_rle)

                pred_mask_path = osp.join(vid_path, exp, f"{frame_name}.png")
                pred_masks[frame_idx] = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

                # foreground mask
                if mask_dict_foreground:
                    mask_fore_rle = mask_dict_foreground[vid_name]["masks_rle"][frame_idx]
                    mask_fore = cocomask.decode(mask_fore_rle)
                    mask_fore = (
                        mask_fore.sum(axis=2).astype(np.uint8)
                        if mask_fore.ndim == 3
                        else mask_fore.astype(np.uint8)
                    )
                    foreground_masks[frame_idx] = mask_fore

            j = db_eval_iou(gt_masks, pred_masks).mean()
            f = db_eval_boundary(gt_masks, pred_masks).mean()
            metrics = [j, f]
            if mask_dict_foreground:
                a = get_r2vos_accuracy(gt_masks, pred_masks).mean()
                r = get_r2vos_robustness(gt_masks, pred_masks, foreground_masks).mean()
                metrics = [*metrics, a, r]
            out_dict[exp_name] = metrics
        except:
            print(colored(f"error: {exp_name}, {traceback.format_exc()}", "red"))
            metrics = [0.0, 0.0]
            if mask_dict_foreground:
                metrics = [*metrics, 0.0, 0.0]
            out_dict[exp_name] = metrics


def parse_revos_results(output_dict):
    data_list = []
    for videxp, (j, f, a, r) in output_dict.items():
        vid_name, exp = videxp.rsplit("_", 1)
        data = {}

        data["video_name"] = vid_name
        data["exp_id"] = exp
        data["exp"] = exp_dict[vid_name]["expressions"][exp]["exp"]
        data["videxp"] = videxp
        data["J"] = round(100 * j, 2)
        data["F"] = round(100 * f, 2)
        data["JF"] = round(100 * (j + f) / 2, 2)
        data["A"] = round(100 * a, 2)
        data["R"] = round(100 * r, 2)
        data["type_id"] = exp_dict[vid_name]["expressions"][exp]["type_id"]

        data_list.append(data)

    is_referring = lambda x: x["type_id"] == 0
    is_reason = lambda x: x["type_id"] == 1
    is_nan = lambda x: x["type_id"] == 2

    j_referring = np.array([d["J"] for d in data_list if is_referring(d)]).mean()
    f_referring = np.array([d["F"] for d in data_list if is_referring(d)]).mean()
    a_referring = np.array([d["A"] for d in data_list if is_referring(d)]).mean()
    r_referring = np.array([d["R"] for d in data_list if is_referring(d)]).mean()
    jf_referring = (j_referring + f_referring) / 2

    j_reason = np.array([d["J"] for d in data_list if is_reason(d)]).mean()
    f_reason = np.array([d["F"] for d in data_list if is_reason(d)]).mean()
    a_reason = np.array([d["A"] for d in data_list if is_reason(d)]).mean()
    r_reason = np.array([d["R"] for d in data_list if is_reason(d)]).mean()
    jf_reason = (j_reason + f_reason) / 2

    j_referring_reason = (j_referring + j_reason) / 2
    f_referring_reason = (f_referring + f_reason) / 2
    a_referring_reason = (a_referring + a_reason) / 2
    r_referring_reason = (r_referring + r_reason) / 2
    jf_referring_reason = (jf_referring + jf_reason) / 2

    results = {
        "referring": {
            "J": j_referring,
            "F": f_referring,
            "A": a_referring,
            "R": r_referring,
            "JF": jf_referring,
        },
        "reason": {"J": j_reason, "F": f_reason, "A": a_reason, "R": r_reason, "JF": jf_reason},
        "overall": {
            "J": j_referring_reason,
            "F": f_referring_reason,
            "A": a_referring_reason,
            "R": r_referring_reason,
            "JF": jf_referring_reason,
        },
    }
    return results, data_list


def parse_refvos_results(output_dict):
    data_list = []
    for videxp, (j, f) in output_dict.items():
        vid_name, exp = videxp.rsplit("_", 1)
        data = {}

        data["video_name"] = vid_name
        data["exp_id"] = exp
        data["exp"] = exp_dict[vid_name]["expressions"][exp]["exp"]
        data["videxp"] = videxp
        data["J"] = round(100 * j, 2)
        data["F"] = round(100 * f, 2)
        data["JF"] = round(100 * (j + f) / 2, 2)

        data_list.append(data)

    j = np.array([d["J"] for d in data_list]).mean()
    f = np.array([d["F"] for d in data_list]).mean()
    jf = (j + f) / 2

    results = {
        "overall": {
            "J": j,
            "F": f,
            "JF": jf,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_path", type=str)
    parser.add_argument("--exp_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--foreground_mask_path", type=str, default="")
    args = parser.parse_args()
    is_davis = "davis" in args.exp_path.lower()

    pred_path = args.pred_path
    pred_path = pred_path if pred_path[-1] != "/" else pred_path[:-1] # if end by separator, remove it
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

    mask_dict_foreground = None
    if args.foreground_mask_path:
        mask_dict_foreground = json.load(open(args.foreground_mask_path, "r"))

    for vid_name in exp_dict:
        vid = exp_dict[vid_name]
        for exp in vid["expressions"]:
            queue.put([vid_name, exp])

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

    if args.foreground_mask_path:  # only for ReVOS
        results, data_list = parse_revos_results(output_dict)
    else:
        results, data_list = parse_refvos_results(output_dict)

    ## save average results
    # eval_dataset, eval_model_cfg = pred_path.split("/")[-2:]
    # SAVE_ROOT = osp.join("metrics", eval_model_cfg)
    # os.makedirs(SAVE_ROOT, exist_ok=True)
    # output_json_path = osp.join(SAVE_ROOT, f"{eval_dataset}.json")
    # output_csv_path = osp.join(SAVE_ROOT, f"{eval_dataset}.csv")
    save_path = pred_path.replace("outputs", "metrics")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output_json_path = save_path + ".json"
    output_csv_path = save_path + ".csv"

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
