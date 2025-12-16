# Decomposed Attention Fusion in MLLMs for Training-Free Video Reasoning Segmentation


[arXiv](https://arxiv.org/abs/2510.19592) | [Project Page](https://www.jshyun.me/projects/decaf)

## Update
* 🗓️ `Coming Soon`: Demo visualization code will be released!
* 🗓️ `Dec 9, 2025`: Code is released!

## 📝 TL;DR

DecAF is explores training-free adaptation of MLLMs for video segmentation (+keypoint localization) through our novel attention-fusion method. While our attention-fusion method enables MLLMs to directly provide coarse video segmentation masks, our attention-guided SAM2 prompting method further allows the generation of dense masks. Our method achieves performance comparable to training-based methods and even surpasses them on ReasonVOS, a benchmark requiring complex reasoning and evaluated in a zero-shot setting.

## ⚙️ Installation
We recommend using a virtual environment (e.g., `venv` or `conda`). We conduct the experiments on A100 GPU with cuda 12.4 version.

```bash
cd {project_root}

# 1. Install PyTorch (CUDA 12.4 build)
pip install torch==2.5.1 torchvision==0.20.1 \
  --extra-index-url https://download.pytorch.org/whl/cu124

# 2. Install this project in editable mode
pip install -e .

# 3. Install flash-attn
pip install flash-attn==2.7.3 --no-build-isolation

# 4. Install SAM 2 (third-party dependency)
cd third_parts/sam2
python -m pip install --no-build-isolation -v -e .
```

## 🗂️ Datasets

We use absolute paths for dataset directories in `evaluation/inf_utils.py`.  
You can modify the following variables and the file paths used in `get_dataset(...)` as needed:

```bash
YTVOS_DATA_ROOT     = "datasets/RVOSJoint/ref-youtube-vos"
DAVIS_DATA_ROOT     = "datasets/RVOSJoint/davis17"
MEVIS_DATA_ROOT     = "datasets/RVOSJoint/mevis"
REVOS_DATA_ROOT     = "datasets/RVOSJoint/ReVOS"
REASONVOS_DATA_ROOT = "datasets/RVOSJoint/ReasonVOS"

# Please organize your datasets following this structure:
{project_root}/datasets/RVOSJoint/
├── davis17
│   └── ...
├── ReasonVOS
│   └── ...
├── ReVOS
│   └── ...
├── mevis              
└── ref-youtube-vos    

# For checkpoints, we assume the following structure:
{project_root}/ckpts/
├── Qwen2.5-VL-7B-Instruct
│   └── ...
└── sam2-hiera-large
    ├── sam2_hiera_l.yaml
    └── sam2_hiera_large.pt
...
sam2-hiera-large
    sam2_hiera_l.yaml
    sam2_hiera_large.pt

# Inference results are saved under:
{project_root}/
├── attn_weights/
│   └── ...
├── outputs/
│   └── ...
└── metrics/
    └── ...
```


For easier setup, we provide tar-compressed files for DAVIS17 and ReasonVOS on Hugging Face:
* https://huggingface.co/datasets/js-hyun/decaf_data

For ReasonVOS, we reformatted the annotations for a unified interface using `data_reformat_reasonvos.py`. You can either:
* Run data_reformat_reasonvos.py on the original ReasonVOS annotations, or
* Directly use the preprocessed ReasonVOS data from the Hugging Face link above.

For ReVOS, mevis, and ref-youtube-vos, due to its large size, please download them from the official repository:
* ReVOS: https://github.com/cilinyan/ReVOS-api


## 📁 File Structure

* Modifications to the MLLM and SAM2 architectures are in `./model`.
* Code to compute and save attention maps is in `./evaluation/inference_attn_map_{mllm_type}.py`.
* Code to obtain segmentation masks is in `./inference_attn_mask.py`.
* Code to obtain keypoints is in `./inference_keypoints.py`.
* Example running scripts are provided in `./evaluation/scripts`.  
  You can adjust the variable `attn_max_batch` depending on your GPU memory. It controls the batch size for frame-wise (image-level) prompting. You can reduce it if you encounter OOM issues, or you can increase it to fully utilize high-end GPU.
* To further reduce memory usage, you can modify the attention rollout implementation to use `bfloat16` instead of `float32`. In our experiments, this only causes a marginal performance drop (~0.x).


## Citation

If you find this project helpful for your research or applications, please cite our paper:

```bibtex
@article{han2025decomposed,
  title={Decomposed Attention Fusion in MLLMs for Training-Free Video Reasoning Segmentation},
  author={Han, Su Ho and Hyun, Jeongseok and Lee, Pilhyeon and Shim, Minho and Wee, Dongyoon and Kim, Seon Joo},
  journal={arXiv preprint arXiv:2510.19592},
  year={2025}
}
```
