# Bridging Modalities via Progressive Re-alignment for Multimodal Test-Time Adaptation
This is the official PyTorch implementation for Bridging Modalities via Progressive Re-alignment for Multimodal Test-Time Adaptation (**AAAI Oral**).

![Overview](./images/Overview.png)

## Prerequisites

```shell
conda create -n mmtta -y python=3.9
conda activate mmtta
pip install -r requirements.txt
```

## Get Started

The corresponding datasets need to be downloaded:

- Kinetics50: Refer to [2024-ICLR-READ](https://github.com/XLearning-SCU/2024-ICLR-READ) to download the training set and validation set.
- VGGSound: Refer to [2024-ICLR-READ](https://github.com/XLearning-SCU/2024-ICLR-READ) to download the testing set. The training set can be downloaded from https://huggingface.co/datasets/Loie/VGGSound.

**Step 1. Introduce corruptions**

```shell
# Video
python ./make_corruptions/make_c_video.py --corruption 'gaussian_noise' --severity 5 --data-path 'data_path/Kinetics50/image_mulframe_val256_k=50' --save_path 'data_path/Kinetics50/image_mulframe_val256_k-C'

# Audio
python ./make_corruptions/make_c_audio.py --corruption 'gaussian_noise' --severity 5 --data_path 'data_path/Kinetics50/audio_val256_k=50' --save_path 'data_path/Kinetics50/audio_val256_k=50-C' --weather_path 'data_path/NoisyAudios/'
```

**Step 2. Create JSON files**

```shell
# JSON file for clean data (**Mandatory**):
python ./data_process/create_clean_json.py --refer-path 'code_path/json_csv_files/ks50_test_refer.json' --video-path 'data_path/Kinetics50/image_mulframe_val256_k=50' --audio-path 'data_path/Kinetics50/audio_val256_k=50' --save_path 'code_path/json_csv_files/ks50' --split 'test'

# JSON file for source data:
python ./data_process/create_clean_json.py --refer-path 'code_path/json_csv_files/ks50_train_refer.json' --video-path 'data_path/Kinetics50/image_mulframe_val256_k=50' --audio-path 'data_path/Kinetics50/audio_val256_k=50' --save_path 'code_path/json_csv_files/ks50' --split 'train'

# JSON file for video-corrupted data:
python ./data_process/create_video_c_json.py --clean-path 'code_path/json_csv_files/ks50/clean/severity_0.json' --video-c-path 'data_path/Kinetics50/image_mulframe_val256_k=50-C' --audio-path 'data_path/Kinetics50/audio_val256_k=50' --corruption 'gaussian_noise'

# JSON file for audio-corrupted data:
python ./data_process/create_audio_c_json.py --clean-path 'code_path/json_csv_files/ks50/clean/severity_0.json' --video-path 'data_path/Kinetics50/image_mulframe_val256_k=50' --audio-c-path 'data_path/Kinetics50/audio_val256_k=50-C' --corruption 'gaussian_noise'

# JSON file for audio & video-corrupted data:
python ./data_process/create_both_c_json.py --clean-path 'code_path/json_csv_files/ks50/clean/severity_0.json' --video-c-path 'data_path/Kinetics50/image_mulframe_val256_k=50-C' --audio-c-path 'data_path/Kinetics50/audio_val256_k=50-C' --dataset 'ks50'
```

## Run Experiments

```shell
# Kinetics50

# unimodal corruption
python run.py --gpu '0, 1, 2' --tta_method BriMPR --corruption_modality [audio/video] --dataset ks50 --json_root 'code_path/json_csv_files/ks50' --label_csv 'code_path/json_csv_files/class_labels_indices_ks50.csv' --pretrain_path 'code_path/pretrained_model/cav_mae_ks50.pth'

# multimodal corruption
python run_both.py --gpu '0, 1, 2' --tta_method BriMPR --corruption_modality both --dataset ks50 --json_root 'code_path/json_csv_files/ks50' --label_csv 'code_path/json_csv_files/class_labels_indices_ks50.csv' --pretrain_path 'code_path/pretrained_model/cav_mae_ks50.pth'
```

```shell
# VGGSound

# unimodal corruption
python run.py --gpu '0, 1, 2' --tta_method BriMPR --corruption_modality [audio/video] --dataset vggsound --json_root 'code_path/json_csv_files/vgg' --label_csv 'code_path/json_csv_files/class_labels_indices_vgg.csv' --pretrain_path 'code_path/pretrained_model/vgg_65.5.pth'

# multimodal corruption
python run_both.py --gpu '0, 1, 2' --tta_method BriMPR --corruption_modality both --dataset vggsound --json_root 'code_path/json_csv_files/vgg' --label_csv 'code_path/json_csv_files/class_labels_indices_vgg.csv' --pretrain_path 'code_path/pretrained_model/vgg_65.5.pth'
```

