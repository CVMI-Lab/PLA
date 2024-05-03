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
#### 3D Semantic Segmentation
- Base-annotated setup

    | Dataset | Partition | hIoU / mIoU(B) / mIoU(N) | Path |
    |:---:|:---:|:---:|:---:|
    | ScanNet | [B15/N4](../tools/cfgs/scannet_models/spconv_clip_base15_caption.yaml) | 69.4 / 68.2 / 70.7 | [ckpt]() |
    | ScanNet | [B12/N7](../tools/cfgs/scannet_models/spconv_clip_base12_caption.yaml) | 68.2 / 69.9 / 66.6 | [ckpt]() |
    | ScanNet | [B10/N9](../tools/cfgs/scannet_models/spconv_clip_base10_caption.yaml) | 64.3 / 76.3 / 55.6 | [ckpt]() |
    | nuScenes | [B12/N3](../tools/cfgs/nuscenes_models/sparseunet_clip_base12_caption.yaml) |  64.4 / 75.8 / 56.0 | [ckpt]() |
    | nuScenes | [B10/N5](../tools/cfgs/nuscenes_models/sparseunet_clip_base10_caption.yaml) |  49.0 / 75.8 / 36.3 | [ckpt]() |
    | ScanNet200 | [B170/N30](../tools/cfgs/scannet200_models/spconv_clip_base170_caption.yaml) | 16.9 / 21.6 / 13.9 | [ckpt]() |
    | ScanNet200 | [B150/N50](../tools/cfgs/scannet200_models/spconv_clip_base150_caption.yaml) | 14.6 / 22.4 / 10.8 | [ckpt]() |

- Annotation-free setup
  
    | Dataset | Model | mIoU  (mAcc) | Path |
    |:---:|:---:|:---:|:---:|
    | ScanNet | [RegionPLC + SparseUNet16]() | 56.9 (75.6) | [ckpt]() |
    | ScanNet | [RegionPLC + SparseUNet32]() | 59.6 (77.5) | [ckpt]() |
    | ScanNet | [RegionPLC + OpenScene + SparseUNet16]() | 60.1 (74.4) | [ckpt]() |
    | ScanNet | [RegionPLC + OpenScene + SparseUNet32]() | 63.6 (80.3)  | [ckpt]() |
    | ScanNet200 | [RegionPLC + SparseUNet32]() |  9.1 (17.3) | [ckpt]() |
    | ScanNet200 | [RegionPLC + OpenScene + SparseUNet32]() |  9.6 (17.8) | [ckpt]() |


#### 3D Instance Segmentation
- Base-annotated setup

    | Dataset | Partition | hAP<sub>50</sub> / mAP<sub>50</sub>(B) / mAP<sub>50</sub>(N) | Path |
    |:---:|:---:|:---:|:---:|
    | ScanNet | [B13/N4](../tools/cfgs/scannet_models/inst/softgroup_clip_base13_caption.yaml) | 58.2 / 59.2 / 57.2 | [ckpt]() |
    | ScanNet | [B10/N7](../tools/cfgs/scannet_models/inst/softgroup_clip_base10_caption.yaml) | 40.6 / 53.9 / 32.5 | [ckpt]() |
    | ScanNet | [B8/N9](../tools/cfgs/scannet_models/inst/softgroup_clip_base8_caption.yaml) | 46.8 / 62.5 / 37.4 | [ckpt]() |
