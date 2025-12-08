
import os
import pickle
import json
import numpy as np
import tqdm
from PIL import Image
from pycocotools import mask as cocomask


meta_filepath = "datasets/RVOSJoint/ReasonVOS/meta_expressions.json"
new_meta_filepath = "datasets/RVOSJoint/ReasonVOS/meta_expressions_v2.json"
mask_root_dir = "datasets/RVOSJoint/ReasonVOS/Annotations"
mask_dict_filepath = "datasets/RVOSJoint/ReasonVOS/mask_dict_v2.pkl"

sorted(os.listdir(mask_root_dir))


with open(meta_filepath, "r") as fp:
    meta = json.load(fp)['videos']

print("Formating meta expression")
anno_ids = []
vids = sorted(list(meta.keys()))
new_meta = {}
for vid in tqdm.tqdm(vids):
    expr_list = meta[vid]['expressions']
    source_dataset = meta[vid]['source']
    new_expr_dict = dict()
    poped_expr_list = []
    for expr in expr_list:
        exp_id = str(expr['exp_id'])
        obj_id = expr['obj_id']
        exp_text = expr['exp_text']
        is_sent = expr['is_sent']
        anno_id = f"{source_dataset}_{vid}_{obj_id}"
        # following ReVOS's multi-object format - id in list
        data = {exp_id: {"exp": exp_text, "obj_id": [obj_id], "anno_id": [anno_id], "is_sent": is_sent}}
        if exp_id in new_expr_dict:
            poped_expr = new_expr_dict.pop(exp_id)
            poped_expr_list.append(poped_expr)

        new_expr_dict.update(data)
        anno_ids.append(anno_id)

        ## handle duplicate expr_id by re-assign expr_id of poped data
        if len(poped_expr_list) > 0:
            curr_expr_id_int = max([int(k) for k in list(new_expr_dict.keys())]) + 1
            while len(poped_expr_list) > 0:
                if str(curr_expr_id_int) not in new_expr_dict:
                    poped_expr = poped_expr_list.pop(0)
                    new_expr_dict.update({str(curr_expr_id_int) : poped_expr})
                curr_expr_id_int += 1
    
    ## replace list of expressions by dict format
    new_meta.update({
        vid: {
            'source': meta[vid]['source'],
            'frames': meta[vid]['frames'],
            'expressions': new_expr_dict,
        }
    })
    # new_meta[vid]['expressions'] = new_expr_dict

## double check the number of expressions
num_expr_ori = sum([len(meta[vid]['expressions']) for vid in vids])
num_expr_new = sum([len(new_meta[vid]['expressions']) for vid in vids])
print(f"Number of expressions - original: {num_expr_ori}, new: {num_expr_new}")

with open(new_meta_filepath, "w") as fp:
    json.dump({'videos': new_meta}, fp)

print("Formating mask dict")
mask_dict = {}
anno_ids = np.unique(anno_ids).tolist()
for anno_id in tqdm.tqdm(anno_ids):
    mask_folder = os.path.join(mask_root_dir, anno_id)
    mask_filenames = sorted(os.listdir(mask_folder))
    mask_rle_list = []
    for mask_filename in mask_filenames:
        mask = np.array(Image.open(os.path.join(mask_folder, mask_filename)))
        mask = (mask > 0).astype(np.uint8)
        mask_rle = cocomask.encode(np.asfortranarray(mask))
        mask_rle_list.append(mask_rle)
    
    mask_dict.update({anno_id: mask_rle_list})

with open(mask_dict_filepath, "wb") as fp:
    pickle.dump(mask_dict, fp)