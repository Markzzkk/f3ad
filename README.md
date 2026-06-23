# CEG5003 Project: Resource Efficient 3D Generative Modeling for Few-Shot Industrial Anomaly Detection

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Quick Start (Demo)](#quick-start)
3. [LDM3D Finetuning](#ldm3d-finetuning)
4. [Anomaly Detector Training and Inference](#train-infer)
   - [Dataset Preparation](#dataset-preparation)
   - [Few-Shot Sampling](#few-shot-sampling)
   - [Model Training](#model-training)
   - [Anomaly Detection / Inference](#anomaly-detection--inference)
   

## Environment Setup

All Python dependencies are listed in `requirements.txt`. We recommend Python ≥ 3.10.

```bash
conda create -n ceg5003 python=3.10
conda activate ceg5003
cd <project_root>
pip install -r requirements.txt
```


## Quick Start
Before we start, please make sure you have the rights to use [DINOv3](https://github.com/facebookresearch/dinov3). Download our trained manifold projectors, and put them to `./logs/`. 
|DINOv3-based|2-shot|
|---------|:---------:|
|**MVTec 3D AD**|[⬇️ <u>link</u>](https://drive.google.com/file/d/11Kh4zPpfwgGePrwLwsGMUD70CfNL3aW5/view?usp=sharing)|
|**Eyecandies**  |[⬇️ <u>link</u>](https://drive.google.com/file/d/1g-eCfet8ccnQYHhT62vmIjmW6xJzqudH/view?usp=drive_link)|


Run a demo on MVTec-3D AD 
```bash
python main.py mode=demo app=test testing.segmentation_vis=True data.dataset=mvtec3d data.data_name=mvtec3d_2shot data.test_root=assets/mvtec3d
```

Or a demo on Eyecandies
```bash
python main.py mode=demo app=test testing.segmentation_vis=True data.dataset=eyecandies data.data_name=eyecandies_2shot data.test_root=assets/eyecandies
```



## LDM3D Finetuning
Before training the anomaly detector, we finetune LDM3D on the few-shot samples to get the anomaly generator. 
Pretrained LDM3D models can be downloaded from [<u>here</u>](https://huggingface.co/Intel/ldm3d). Then, run the following command to finetune LDM3D:
```bash
# Finetune LDM3D on MVTec 3D-AD
python ldm3d_finetune/train_universal.py --dataset_type mvtec3d --data_root /path/to/your/mvtec3d/folder --model /path/to/pretrained/ldm3d --out ./lora_outputs/

# Finetune LDM3D on Eyecandies
python ldm3d_finetune/train_universal.py --dataset_type eyecandies --data_root /path/to/your/eyecandies/folder --model /path/to/pretrained/ldm3d --out ./lora_outputs/
```
where `dataset_type` is either "mvtec3d" or "eyecandies", `data_root` is the finetune dataset folder, and `model` is the pretrained LDM3D checkpoint. After training, the finetuned model will be saved to `./lora_outputs/ldm3d_lora_{dataset_type}` by default. You can specify `--out` to change the saving path.
Note: We use the LoRA finetuned on the other domain as the anomaly generator (cross-domain transfer). When training the detector for MVTec 3D-AD, load the LoRA finetuned on Eyecandies, and vice versa.


## Training and Inference

### Dataset Preparation

| Dataset | Preferred download |
|---------|--------------------|
| **MVTec 3D AD** | Official site: [<u>MVTec 3D AD</u>](https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad) |
| **Eyecandies** | Official site: [<u>Eyecandies</u>](https://eyecan-ai.github.io/eyecandies/). |

After downloading zip files of eyecandies, run the commands below:

```bash
mkdir -p /path/to/f3ad/Eyecandies

for file in /你的下载目录/*.zip; do

    unzip "$file" -d /path/to/f3ad/Eyecandies
done
```

### Few-Shot Sampling

Create a **few-shot** subset with `sample.py`:

```bash
python src/sample.py dataset=mvtec3d source=/path/to/your/dataset target=/path/to/target/folder seed=42 num_samples=2

python src/sample.py dataset=eyecandies source=/path/to/your/dataset target=/path/to/target/folder seed=42 num_samples=2
```
where `dataset` is either "mvtec3d" or "eyecandies", `source` is the dataset folder, `target` is the folder of few-shot samples, and `num_samples` is the number of samples training models, e.g., 2 for 2-shot learning. `seed` can be adjusted to have multiple rounds of experiment.

### Model Training

```bash
# Train the Anomaly Detection Model on MVTec 3D-AD
python main.py  mode=train diy_name=dbug  data.dataset=mvtec3d   data.use_depth=true data.depth_repr=normal  data.data_path=/path/to/your/few-shot/folder   data.data_name=mvtec3d_2shot   data.test_root=/path/to/your/mvtec3d/folder   anomaly_synth.type=ldm3d   anomaly_synth.ldm3d.model_path=/path/to/your/pretrained/ldm3d   anomaly_synth.ldm3d.cross_domain_lora.mvtec3d=./lora_outputs/ldm3d_lora_eyecandies/final/unet_lora.pth anomaly_synth.ldm3d.dataset_roots.mvtec3d=/path/to/your/mvtec3d/folder   anomaly_synth.ldm3d.cache_root=./anomaly_cache/mvtec3d_2shot  anomaly_synth.ldm3d.cache_k=2000

# Train the Anomaly Detection Model on Eyecandies
python main.py  mode=train diy_name=dbug  data.dataset=eyecandies   data.use_depth=true data.depth_repr=normal  data.data_path=/path/to/your/few-shot/folder   data.data_name=eyecandies_2shot   data.test_root=/path/to/your/eyecandies/folder   anomaly_synth.type=ldm3d   anomaly_synth.ldm3d.model_path=/path/to/your/pretrained/ldm3d   anomaly_synth.ldm3d.cross_domain_lora.eyecandies=./lora_outputs/ldm3d_lora_mvtec3d/final/unet_lora.pth anomaly_synth.ldm3d.dataset_roots.eyecandies=/path/to/your/eyecandies/folder   anomaly_synth.ldm3d.cache_root=./anomaly_cache/eyecandies_2shot  anomaly_synth.ldm3d.cache_k=2000
```
where `diy_name`is the post-fix name of the model saving directory, `data.dataset` is either "mvtec3d" or "eyecandies", `data.data_path` is the path where the few-shot folder is at, and `data.test_root` is the original dataset folder for testing. `anomaly_synth.ldm3d.model_path` is the pretrained LDM3D checkpoint, and `anomaly_synth.ldm3d.cross_domain_lora` is the finetuned LDM3D LoRA checkpoint on the other dataset. `anomaly_synth.ldm3d.dataset_roots` is the original dataset folder used for mask augmentation, and `anomaly_synth.ldm3d.cache_root` is the folder for caching synthesized anomalies. `anomaly_synth.ldm3d.cache_k` is the number of synthesized anomalies to cache, recommended equal to the number of training epochs.


### Anomaly Detection / Inference

After training, run inference:

```bash
# Test the Anomaly Detection Model on MVTec 3D-AD
python main.py mode=AD app=test diy_name=dbug data.dataset=mvtec3d data.data_name=mvtec3d_2shot data.use_depth=true data.depth_repr=normal data.test_root=/path/to/your/mvtec3d/folder app.ckpt_step=4000

# Test the Anomaly Detection Model on Eyecandies
python main.py mode=AD app=test diy_name=dbug data.dataset=eyecandies data.data_name=eyecandies_2shot data.use_depth=true data.depth_repr=normal data.test_root=/path/to/your/eyecandies/folder app.ckpt_step=4000
```
where `data.test_root` is the dataset folder, either "mvtec3d" or "eyecandies", and `app.ckpt_step` is the checkpoint step to load for inference, which can be adjusted based on the training.

