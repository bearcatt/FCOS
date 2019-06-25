# High-resolution Networks for FCOS

## Introduction
This project contains the code of HRNet-FCOS, i.e., using [High-resolution Networks (HRNets)](https://arxiv.org/pdf/1904.04514.pdf) as the backbones for the [Fully Convolutional One-Stage Object Detection (FCOS)](https://arxiv.org/abs/1904.01355) algorithm, which achieves much better detection results compared with the ResNet-FCOS counterparts while keeping a similar computation complexity. For more projects using HRNet, please go to our [website](https://github.com/HRNet).

## Quick start
### Installation

Please check [INSTALL.md](INSTALL.md) for installation instructions.
You may also want to see the original [README.md](FCOS_README.md) of FCOS.

### Inference
The inference command line on coco minival split:

    python tools/test_net.py \
        --config-file configs/fcos/fcos_hrnet_w32_5l_2x.yaml \
        MODEL.WEIGHT models/FCOS_hrnet_w32_5l_2x.pth \
        TEST.IMS_PER_BATCH 8

Please note that:
1) If your model's name is different, please replace `models/FCOS_hrnet_w32_5l_2x.pth` with your own.
2) If you enounter out-of-memory error, please try to reduce `TEST.IMS_PER_BATCH` to 1.
3) If you want to evaluate a different model, please change `--config-file` to its config file (in [configs/fcos](configs/fcos)) and `MODEL.WEIGHT` to its weights file.

For your convenience, we provide the following trained models.

Model | Training mem (GB) | Multi-scale training | SyncBN| Testing time / im | Backbone GFLOPs| AP (minival) | AP (test-dev) | Link
--- |:---:|:---:|:---:|:---:|:---:|:---:|:--:|:---:
FCOS_R_50_5l_2x                | 29.3 | No  |No | 71ms  |90.6   | 37.1 | -    | [download]()
FCOS_HRNet_W18_5l_2x           | 54.4 | No  |No | 75ms  |80.6  | 37.7 | -    | [download]()
FCOS_HRNet_W18_5l_2x           | 54.4 | Yes |Yes| 75ms  |80.6  | -    | -    | [download]()
|
FCOS_R_50_6l_2x                | 58.2 | No  |No | 75ms  |130.5  | 37.1 | -    | [download]()
FCOS_HRNet_W18_6l_2x           | 88.1 | No  |No | 105ms |116.5 | 37.8 | -    | [download]()
FCOS_HRNet_W18_6l_2x           | 88.1 | Yes |Yes| 105ms |116.5 | -    | -    | [download]()
|
FCOS_R_101_5l_2x               | 44.1 | Yes |No | 74ms  |162.8  | 41.4 | -    | [download]()
FCOS_HRNet_W32_5l_2x           | 78.9 | Yes |No | 82ms  |173.6 | 41.9 | -    | [download]()
FCOS_HRNet_W32_5l_2x           | 78.9 | Yes |Yes| 82ms  |173.6 | -    | -    | [download]()
|
FCOS_R_101_6l_2x               | 71.0 | Yes |No | 115ms |202.7  | 41.5 | -    | [download]()
FCOS_HRNet_W32_6l_2x           | 108.6| Yes |No | 120ms |209.5 | 42.1 | -    | [download]()
FCOS_HRNet_W32_6l_2x           | 108.6| Yes |Yes| 120ms |209.5 | 43.0    | -    | [download]()
|
FCOS_HRNet_W40_6l_3x           | 128.0| Yes |No | 139ms |284.4 | 42.6 | -    | [download]()

[1] *1x and 2x mean the model is trained for 90K and 180K iterations, respectively.* \
[2] *We report total training memory footprint on all GPUs instead of the memory footprint per GPU as in maskrcnn-benchmark.* \
[3] *The branches in HRNet model cannot run in parallel since Pytorch adopts dynamic computation graph, which leading to a slower inference speed than ResNet.* \
[5] *We provide HRNet-FCOS models trained with Synchronous Batch-Normalization (syncBN).*\
[6] *5l and 6l denote that we use feature pyramid with 5 levels and 6 levels, respectively.*\
[4] *All results are obtained with a single model and without any test time data augmentation.*

HRNet 

### Training

The following command line will trains a fcos_hrnet_w32_5l_2x model on 8 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --config-file configs/fcos/fcos_hrnet_w32_5l_2x.yaml \
        MODEL.WEIGHT hrnetv2_w32_imagenet_pretrained.pth \
        DATALOADER.NUM_WORKERS 4 \
        OUTPUT_DIR training_dir/fcos_hrnet_w32_5l_2x
        
Note that:
1) If you want to use fewer GPUs, please change `--nproc_per_node` to the number of GPUs. No other settings need to be changed. The total batch size does not depends on `nproc_per_node`. If you want to change the total batch size, please change `SOLVER.IMS_PER_BATCH` in [configs/fcos/fcos_hrnet_w32_5l_2x.yaml](configs/fcos/fcos_hrnet_w32_5l_2x.yaml).
2) The imagenet pre-trained model can be found [here](https://github.com/HRNet/HRNet-Object-Detection#faster-r-cnn).
3) The models will be saved into `OUTPUT_DIR`.
4) If you want to train FCOS on your own dataset, please follow this instruction [#54](https://github.com/tianzhi0549/FCOS/issues/54#issuecomment-497558687).
### Contributing to the project

Any pull requests or issues are welcome.

### Citations
Please consider citing the following papers in your publications if the project helps your research. 
```
@article{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  journal={arXiv preprint arXiv:1902.09212},
  year={2019}
}

@article{tian2019fcos,
  title   =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author  =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  journal =  {arXiv preprint arXiv:1904.01355},
  year    =  {2019}
}
```


### License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 
