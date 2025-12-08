##### Set Config
## Model
MODEL_NAME_INTERNVL_2B="intern-vl-2b-rvos"
VERSION_INTERNVL_2B="ckpts/InternVL3-2B-hf"
MODEL_NAME_INTERNVL_8B="intern-vl-8b-rvos"
VERSION_INTERNVL_8B="ckpts/InternVL3-8B-hf"
MODEL_NAME_INTERNVL_14B="intern-vl-14b-rvos"
VERSION_INTERNVL_14B="ckpts/InternVL3-14B-hf"

# Select model and version
MODEL_NAME=${MODEL_NAME_INTERNVL_14B}
VERSION=${VERSION_INTERNVL_14B}

## Data
DATA_NAME_REVOS="revos_valid"
DATA_CFG_REVOS="--dataset revos"
EVAL_DATA_CFG_REVOS="--exp_path datasets/RVOSJoint/ReVOS/meta_expressions_valid_.json \
                --mask_path datasets/RVOSJoint/ReVOS/mask_dict.json \
                --foreground_mask_path datasets/RVOSJoint/ReVOS/mask_dict_foreground.json"
DATA_NAME_DAVIS="davis_valid"
DATA_CFG_DAVIS="--dataset davis"
EVAL_DATA_CFG_DAVIS="--exp_path datasets/RVOSJoint/davis17/meta_expressions/valid/meta_expressions.json \
            --mask_path datasets/RVOSJoint/davis17/valid/mask_dict.pkl"
DATA_NAME_REASONVOS="reasonvos_valid"
DATA_CFG_REASONVOS="--dataset reasonvos"
EVAL_DATA_CFG_REASONVOS="--exp_path datasets/RVOSJoint/ReasonVOS/meta_expressions_v2.json \
            --mask_path datasets/RVOSJoint/ReasonVOS/mask_dict_v2.pkl"

# Select dataset and config
DATA_NAME=${DATA_NAME_REVOS}
DATA_CFG=${DATA_CFG_REVOS}
EVAL_DATA_CFG=${EVAL_DATA_CFG_REVOS}

## Attention Score
ATTN_NUM_FRAME=16
attn_layer_idx_start=24
attn_layer_idx_end=-1 # -1 means the last layer
ATTN_MS_CFG="--attn_max_batch 4 --max_patches 12"

ATTN_NAME="F${ATTN_NUM_FRAME}_L${attn_layer_idx_start}-${attn_layer_idx_end}"
ATTN_SCORE_FOLDER="./attn_weights/${MODEL_NAME}/${ATTN_NAME}/${DATA_NAME}"
ATTN_SCORE_CFG="--num_input_frames=${ATTN_NUM_FRAME} --attn_save_path=${ATTN_SCORE_FOLDER} --attn_layer_idx_start ${attn_layer_idx_start}"
## Mask Inference
MASK_NUM_FRAME=16
INF_NAME="inf-uniform-F${MASK_NUM_FRAME}"
MASK_CFG="--num_input_frames=${MASK_NUM_FRAME} --attn_save_path=${ATTN_SCORE_FOLDER} --use_saved_attn"
ATTN_MASK_CFG="${MASK_CFG} --predict_attn_mask"
##### Set Config

# ##### Save attention weights
GPU_LIST=(7)
SUBSET_NUM=${#GPU_LIST[@]}
for ((i=0; i<$SUBSET_NUM; i++)); do
    CUDA_VISIBLE_DEVICES=${GPU_LIST[i]} PYTHONPATH=$(pwd) python evaluation/inference_attn_map_internvl.py \
    --subset_num=${SUBSET_NUM} --subset_idx=$i --version=$VERSION ${DATA_CFG} ${ATTN_SCORE_CFG} ${ATTN_MS_CFG} &
done
wait
echo "(${DATA_NAME}) All Completed for ATTN score"
##### Save attention weights

##### Obtain ATTN mask
GPU_LIST=(7 7)
SUBSET_NUM=${#GPU_LIST[@]}
ATTN_MASK_FOLDER="./outputs/${MODEL_NAME}/${ATTN_NAME}/${INF_NAME}-attn-mask/${DATA_NAME}"
echo "▶ Starting inference for ATTN mask..."
for ((i=0; i<$SUBSET_NUM; i++)); do
    CUDA_VISIBLE_DEVICES=${GPU_LIST[i]} PYTHONPATH=$(pwd) python evaluation/inference_attn_mask.py \
    --subset_num=${SUBSET_NUM} --subset_idx=$i --version=$VERSION \
    --vis_save_path=${ATTN_MASK_FOLDER} ${DATA_CFG} ${ATTN_MASK_CFG} &
done
wait
echo "(${DATA_NAME}) All Completed for ATTN mask"
python evaluation/eval_rvos_obj_iou.py ${ATTN_MASK_FOLDER} ${EVAL_DATA_CFG}
echo Evaluate ${ATTN_MASK_FOLDER}
#### Obtain ATTN mask

##### Obtain SAM masks
GPU_LIST=(7 7)
SUBSET_NUM=${#GPU_LIST[@]}
SCORE_THR=0.80
POINT_THR=0.80
SAM_MASK_CFG="${MASK_CFG} --predict_sam_mask --final_score_threshold=${SCORE_THR} --point_threshold=${POINT_THR}"
SAM_MASK_FOLDER="./outputs/${MODEL_NAME}/${ATTN_NAME}/${INF_NAME}-sam-mask/${DATA_NAME}"
echo "▶ Starting inference for SCORE_THR=${SCORE_THR} | POINT_THR=${POINT_THR}..."
for ((i=0; i<$SUBSET_NUM; i++)); do
CUDA_VISIBLE_DEVICES=${GPU_LIST[i]} PYTHONPATH=$(pwd) python evaluation/inference_attn_mask.py \
    --subset_num=${SUBSET_NUM} --subset_idx=$i --version=$VERSION \
    --vis_save_path=${SAM_MASK_FOLDER} ${DATA_CFG} ${SAM_MASK_CFG} &
done
wait
echo "(${DATA_NAME}) All Completed for SCORE_THR=${SCORE_THR} & POINT_THR=${POINT_THR}..."
python evaluation/eval_rvos.py ${SAM_MASK_FOLDER} ${EVAL_DATA_CFG}
echo Evaluate ${SAM_MASK_FOLDER}