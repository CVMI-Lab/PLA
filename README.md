# PLA (Point-Language Association)
**Language-Driven Open-Vocabulary 3D Scene Understanding**

![framwork](./docs/framework.png)
![association](./docs/association_module.png)

**Authors**: Runyu Ding\*, Jihan Yang\*, Chuhui Xue, Wenqing Zhang, Song Bai, Xiaojuan Qi  (\*equal contribution)

[project page](https://dingry.github.io/projects/PLA) | [arXiv](https://arxiv.org/abs/2211.16312)

### TODO
- [ ] Release code and pretrained model (soon)

### Getting Started

#### Installation
Please refer to [INSTALL.md](docs/INSTALL.md) for the installation.

#### Dataset Preparation
Please refer to [DATASET.md](docs/INSTALL.md) for dataset preparation.

#### Training

```bash
cd tools
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} ${PY_ARGS}
```

For instance,
- train B15/N4 semantic segmentation on ScanNet:
    ```bash
    cd tools
    sh scripts/dist_train.sh 8 --cfg_file cfgs/scannet_models/spconv_clip_base15_caption_adamw.yaml --extra_tag exp_tag
    ```
- train B13/N4 instance segmentation on ScanNet:
    ```bash
    cd tools
    sh scripts/dist_train.sh 8 --cfg_file cfgs/scannet_models/inst/softgroup_clip_base13_caption_adamw.yaml --extra_tag exp_tag
    ```

#### Inference

```bash
cd tools
sh scripts/dist_test.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --ckpt ${CKPT_PATH}
```

For instance,
- to test a B15/N4 model on ScanNet:
    ```bash
    cd tools
    sh scripts/dist_test.sh 8 --cfg_file cfgs/scannet_models/spconv_clip_base15_caption_adamw.yaml --ckpt output/scannet_models/spconv_clip_base15_caption/exp_tag/ckpt/checkpoint_ep128.pth
    ```

### Model Zoo
- semantic segmentation

    | Dataset | Partition | hIoU / mIoU(B) / mIoU(N) | Path |
    |:---:|:---:|:---:|:---:|
    | ScanNet | B15/N4 |||
    | ScanNet | B12/N7 |||
    | ScanNet | B10/N9 |||
    | S3DIS | B8/N4 |||
    | S3DIS | B6/N6 |||


- instance segmentation

    | Dataset | Partition | hAP$_{50}$ / hAP$_{50}$(B) / hAP$_{50}$ (N) | Path |
    |:---:|:---:|:---:|:---:|
    | ScanNet | B13/N4 |||
    | ScanNet | B10/N7 |||
    | ScanNet | B8/N9 |||
    | S3DIS | B8/N4 |||
    | S3DIS | B6/N6 |||


### Citation
If you find this project useful in your research, please consider cite:
```bibtex
@inproceedings{ding2022language,
    title={PLA: Language-Driven Open-Vocabulary 3D Scene Understanding},
    author={Ding, Runyu and Yang, Jihan and Xue, Chuhui and Zhang, Wenqing and Bai, Song and Qi, Xiaojuan},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2023}
}
```

### Acknowledgement
Code is partly borrowed from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), [PointGroup](https://github.com/dvlab-research/PointGroup) and [SoftGroup](https://github.com/thangvubk/SoftGroup).