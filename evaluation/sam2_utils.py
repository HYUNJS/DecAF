import torch
import torch.nn.functional as F
import numpy as np

def compute_mask_iou(mask1, mask2):
    """mask1, mask2: [H, W] (binary mask)"""
    assert mask1.shape == mask2.shape, "Masks must have the same shape"
    intersection = (mask1 & mask2).float().sum()
    union = (mask1 | mask2).float().sum()
    if union == 0:
        return torch.tensor(0.0, device=mask1.device)
    return intersection / union

def get_valid_obj_ids(
    masks, scores, iou_threshold=0.7, previous_masks=None, previous_scores=None
):
    """
    Get valid object ids based on masks and scores.
    Args:
        masks: Tensor of shape (N, 1, H, W) representing masks. (binary masks)
        scores: Tensor of shape (N,) representing scores.
        iou_threshold: Threshold for IoU to consider an object valid.
        previous_masks: Optional Tensor of shape (M, 1, H, W) representing previous masks for IoU calculation. (binary masks)
        previous_scores: Optional Tensor of shape (M,) representing scores of previous masks.
    Returns:
        List of valid object ids. Optionally, list of delete object ids of previous masks.
    """
    N = masks.shape[0]
    masks = masks.squeeze(1)  # Convert to shape (N, H, W)
    sorted_scores, indices = scores.sort(descending=True)
    masks = masks[indices]  # Sort masks according to scores

    keep = []
    suppressed = torch.zeros(N, dtype=torch.bool, device=masks.device)
    delete_prev_indices = []

    for i in range(N):
        if suppressed[i]:
            continue
        keep.append(indices[i].item())
        for j in range(i + 1, N):
            if suppressed[j]:
                continue
            iou = compute_mask_iou(masks[i], masks[j])
            if iou > iou_threshold:
                suppressed[j] = True

        if previous_masks is not None:
            assert previous_scores is not None
            previous_masks = previous_masks.squeeze(1)  # Convert to shape (M, H, W)
            M = previous_masks.shape[0]
            score_i = sorted_scores[i]
            for k in range(M):
                iou = compute_mask_iou(masks[i], previous_masks[k].bool())
                if iou > iou_threshold:
                    if score_i > previous_scores[k]:
                        delete_prev_indices.append(k)
                    else:
                        # delete the current object idx from keep
                        keep.pop()
                        break

    return torch.tensor(keep, device=masks.device), delete_prev_indices


def get_mask_tracklet_from_point(
    predictor,
    inference_state,
    points_prompt,
    labels_prompt,
    frame_idx,
    attn_score,
    ori_size,
    mask_tracklet_dict_list=None,
):
    """
    Get mask tracklet from point prompt.
    Args:
        inference_state: dict, the inference state containing the model and other info.
        points_prompt: Tensor of shape (N, 1, 2) representing point coordinates.
        labels_prompt: Tensor of shape (N, 1) representing point labels.
        frame_idx: int, index of the frame to process.
        attn_score: Tensor of shape (N, ) representing attention score for the frame.
        ori_size: tuple, original size of the frame (height, width).
        reverse: bool, whether propagate forward or backward.
        mask_tracklet_dict_list: list of dicts, each dict contains 'masks' and 'scores' for a tracklet.
    """

    # points_prompt: [N, 1, 2]
    # labels_prompt: [N, 1]
    if points_prompt is not None:
        # num_frame = len(mask_prompt)
        num_obj = points_prompt.shape[0]
        mask_out = []
    else:
        assert len(inference_state["output_dict"]["cond_frame_outputs"]) > 0
    ious_list = []
    delete_prev_indices = []
    keep_obj_scores = None
    # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    if points_prompt is not None:
        for obj_idx in range(num_obj):
            _, _, frame_mask_out, ious = predictor.add_new_points_or_box(
                inference_state,
                frame_idx,
                100 + obj_idx,
                points=points_prompt[obj_idx],
                labels=labels_prompt[obj_idx],
                normalize_coords=False,
            )
            ious_list.append(ious)
        ious_list = torch.cat(ious_list, dim=0).squeeze()  # (num_obj,)
        # frame_mask_out: (num_obj, 1, ori_h, ori_w)
        # attn_score: (num_obj, )
        # frame_mask_out = F.interpolate(
        #     frame_mask_out,
        #     size=ori_size,
        #     mode="bilinear",
        #     align_corners=False,
        # )
        frame_binary_mask = frame_mask_out.sigmoid() > 0.5
        obj_scores = ious_list + attn_score
        if len(mask_tracklet_dict_list) > 0:
            previous_masks = []
            previous_scores = []
            for mask_tracklet_dict in mask_tracklet_dict_list:
                previous_masks.append(
                    mask_tracklet_dict["mask_tracklet"][frame_idx : frame_idx + 1]
                )  # (1, H, W)
                previous_scores.append(mask_tracklet_dict["score"])
            previous_masks = torch.stack(previous_masks, dim=0)  # (prev_num_obj, 1, H, W)
            previous_scores = torch.stack(previous_scores, dim=0)  # (prev_num_obj, )
        else:
            previous_masks = None
            previous_scores = None
        keep_obj_ids, delete_prev_indices = get_valid_obj_ids(
            frame_binary_mask,
            obj_scores,
            iou_threshold=0.7,
            previous_masks=previous_masks,
            previous_scores=previous_scores,
        )
        if len(keep_obj_ids) == 0:
            return None, None, None
        mask_prompt = frame_binary_mask[keep_obj_ids]  # (num_keep_obj, 1, H, W)
        keep_obj_scores = obj_scores[keep_obj_ids]  # (num_keep_obj, )
        predictor.reset_state(inference_state)

        mask_prompt = mask_prompt.squeeze(1)  # (num_keep_obj, H, W)
        for i in range(len(mask_prompt)):
            _mask_prompt = mask_prompt[i]
            _, _, frame_mask_out = predictor.add_new_mask(
                inference_state,
                frame_idx,
                i + 100,
                _mask_prompt,
            )

    mask_out = []
    mask_out_forward = []
    mask_out_backward = []
    num_video_frames = inference_state["num_frames"]

    # forward pass
    start_frame_idx = frame_idx
    max_frame_num_to_track = num_video_frames - frame_idx
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx, max_frame_num_to_track, reverse=False
    ):
        mask_out_forward.append(out_mask_logits)
    assert len(mask_out_forward) == max_frame_num_to_track
    mask_out_forward = torch.cat(mask_out_forward, dim=1)

    if max_frame_num_to_track == num_video_frames:
        return mask_out_forward, keep_obj_scores, delete_prev_indices
    
    # backward pass
    start_frame_idx = frame_idx - 1
    max_frame_num_to_track = frame_idx
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx, max_frame_num_to_track, reverse=True
    ):
        mask_out_backward.append(out_mask_logits)
    assert len(mask_out_backward) == max_frame_num_to_track
    mask_out_backward = torch.cat(mask_out_backward, dim=1)
    mask_out_backward = torch.flip(mask_out_backward, [1])
    mask_out = torch.cat([mask_out_backward, mask_out_forward], dim=1)

    assert mask_out.shape[1] == num_video_frames

    return mask_out, keep_obj_scores, delete_prev_indices


def get_video_mask_from_mask(
        predictor,
        inference_state,
        mask_prompt,
        frame_idx,
):
    num_video_frames = inference_state["num_frames"]
    num_obj = len(mask_prompt)
    mask_out = []

    for obj_idx in range(num_obj):
        frame_mask_out = []

        _mask_prompt = mask_prompt[obj_idx]
        # frame_idx = frame_indices[obj_idx]
        _, _, frame_mask_out = predictor.add_new_mask(
            inference_state,
            frame_idx,
            obj_idx + 100,
            _mask_prompt,
        )

        mask_out.append(frame_mask_out)

    # forward pass
    mask_out_forward = []
    start_frame_idx = frame_idx
    max_frame_num_to_track = num_video_frames - frame_idx
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx, max_frame_num_to_track, False
    ):
        mask_out_forward.append(out_mask_logits)
    assert len(mask_out_forward) == max_frame_num_to_track
    mask_out_forward = torch.cat(mask_out_forward, dim=1)  # (num_obj, t1, h, w)

    # backward pass
    if max_frame_num_to_track == num_video_frames:
        return mask_out_forward

    # backward pass
    mask_out_backward = []
    start_frame_idx = frame_idx - 1
    max_frame_num_to_track = frame_idx
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx, max_frame_num_to_track, True
    ):
        mask_out_backward.append(out_mask_logits)
    assert len(mask_out_backward) == max_frame_num_to_track
    mask_out_backward = torch.cat(mask_out_backward, dim=1)  # (num_obj, t2, h, w)
    mask_out_backward = torch.flip(mask_out_backward, [1])  # flip the time dimension

    all_pred_masks = torch.cat(
        [mask_out_backward, mask_out_forward], dim=1
    )  # (num_obj, t1 + t2, h, w)
    assert all_pred_masks.shape[1] == num_video_frames

    return all_pred_masks

    

def get_AC_score(
    nor_attn_weights,
    mask_tracklet_dict_list,
):
    """
    Get Attention Consistency score for each mask tracklet based on attention weights.
    Args:
        nor_attn_weights: Tensor of shape (T_s, H_p, W_p) representing normalized attention weights for N objects.
        mask_tracklet_dict_list: list of dicts, each dict contains 'masks' and 'scores' for a tracklet.
    Returns:
        Tensor of shape (N,) representing AC scores for each object.
    """

    _, grid_h, grid_w = nor_attn_weights.shape
    
    threshold = nor_attn_weights.mean(dim=(1, 2), keepdim=True)  # (grid_t, 1, 1)
    binary_mask = (nor_attn_weights > threshold).float()  # (grid_t, grid_h, grid_w)

    denominator = nor_attn_weights[binary_mask.bool()].sum()
    grid_scores = nor_attn_weights * binary_mask  # (grid_t, grid_h, grid_w)

    # Set the penalty score on the background grid cells to max value each grid_t
    max_val_per_t = grid_scores.amax(dim=(1, 2), keepdim=True)  # (grid_t, 1, 1)
    grid_scores = torch.where(binary_mask == 0, max_val_per_t * -1.0, grid_scores)

    ac_scores = []

    for i in range(len(mask_tracklet_dict_list)):
        mask_tracklet_dict = mask_tracklet_dict_list[i]
        mask_tracklet = mask_tracklet_dict["mask_tracklet"]  # (T, H, W)

        mask_down = F.adaptive_avg_pool2d(mask_tracklet.unsqueeze(0), output_size=(grid_h, grid_w)).squeeze(0)
        # (T, grid_h, grid_w)
        mask_attention_weighted = mask_down * grid_scores  # (T, grid_h, grid_w)

        numerator = mask_attention_weighted.sum()
        ac_score = numerator / (denominator + 1e-8)
        ac_scores.append(ac_score)

    ac_scores = torch.stack(ac_scores, dim=0)  # (valid_num_obj, )
    pos_scores_mask = ac_scores > 0
    positive_scores = ac_scores[pos_scores_mask]
    if positive_scores.numel() > 0:
        max_val = positive_scores.max()
        ac_scores[pos_scores_mask] = positive_scores / (max_val + 1e-8)

    return ac_scores


def get_final_obj_prompts(
    mask_tracklet_dict_list,
    ac_scores,
    key_frame_indices,
    final_score_threshold=0.8,
):
    mask_prompt = []
    mask_cond_frame_indices = []
    temp_final_scores_list = []

    for i in range(len(mask_tracklet_dict_list)):
        mask_tracklet_dict = mask_tracklet_dict_list[i]
        mask_tracklet = mask_tracklet_dict["mask_tracklet"]  # (T, H, W)
        score = mask_tracklet_dict["score"]
        local_t = mask_tracklet_dict["frame_idx"]  # int
        final_score = (score + ac_scores[i]) / 3.0
        temp_final_scores_list.append(final_score.detach().cpu().item())
        if final_score > final_score_threshold:
            mask_prompt.append(mask_tracklet[local_t].unsqueeze(0))  # (1, H, W)
            mask_cond_frame_indices.append(key_frame_indices[local_t])

    if len(mask_prompt) == 0:
        max_idx = np.argmax(temp_final_scores_list)
        mask_tracklet_dict = mask_tracklet_dict_list[max_idx]
        mask_tracklet = mask_tracklet_dict["mask_tracklet"]  # (T, H, W)
        score = mask_tracklet_dict["score"]
        local_t = mask_tracklet_dict["frame_idx"]  # int
        mask_prompt.append(mask_tracklet[local_t].unsqueeze(0))  # (1, H, W)
        mask_cond_frame_indices.append(key_frame_indices[local_t])

    return mask_prompt, mask_cond_frame_indices