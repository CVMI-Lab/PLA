#### Training

```bash
cd tools
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} ${PY_ARGS}
```

For instance,
- train B15/N4 semantic segmentation on ScanNet:
    ```bash
    cd tools
    sh scripts/dist_train.sh 8 --cfg_file cfgs/scannet_models/spconv_clip_base15_caption.yaml --extra_tag exp_tag
    ```
- train B13/N4 instance segmentation on ScanNet:
    ```bash
    cd tools
    sh scripts/dist_train.sh 8 --cfg_file cfgs/scannet_models/inst/softgroup_clip_base13_caption.yaml --extra_tag exp_tag
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
    sh scripts/dist_test.sh 8 --cfg_file cfgs/scannet_models/spconv_clip_base15_caption.yaml --ckpt output/scannet_models/spconv_clip_base15_caption/exp_tag/ckpt/checkpoint_ep128.pth
    ```

### Model Zoo
#### 3D Semantic Segmentation
- Base-annotated setup

    | Dataset | Partition | hIoU / mIoU(B) / mIoU(N) | Path |
    |:---:|:---:|:---:|:---:|
    | ScanNet | [B15/N4](../tools/cfgs/scannet_models/spconv_clip_base15_caption.yaml) | 69.4 / 68.2 / 70.7 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EWGG39VOw2NOvKdfUjWnh8UBPgU6zBilXLaKGMgO2asBYw?e=g7QCpC) |
    | ScanNet | [B12/N7](../tools/cfgs/scannet_models/spconv_clip_base12_caption.yaml) | 68.2 / 69.9 / 66.6 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EfJpjpul0cBAhHhV3oKm6JkB0UcTUf7TnaLWDgwWkFFGxg?e=OT0wsu) |
    | ScanNet | [B10/N9](../tools/cfgs/scannet_models/spconv_clip_base10_caption.yaml) | 64.3 / 76.3 / 55.6 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EZG4gIqTA2lHpephZLXG8tsBQjA-m5e0HXK_ykjaNK_saQ?e=s6q2Dk) |
    | nuScenes | [B12/N3](../tools/cfgs/nuscenes_models/sparseunet_clip_base12_caption.yaml) |  64.4 / 75.8 / 56.0 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/Ef7cm0XNBjZHupbpYvg9ItkBC8WaB25Ar0kiOTdg5ezz3w?e=dM7Dq6) |
    | nuScenes | [B10/N5](../tools/cfgs/nuscenes_models/sparseunet_clip_base10_caption.yaml) |  49.0 / 75.8 / 36.3 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EW3r1oYwaERDk7BE2v9mp4MB0JOBkYMPKWYIWNkbl_EWGQ?e=cHb1g0) |
    | ScanNet200 | [B170/N30](../tools/cfgs/scannet200_models/spconv_clip_base170_caption.yaml) | 16.9 / 21.6 / 13.9 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EY6yrNOeKSJEo-526WrcOesBZREtteoZ1KBIGPI26Wq2UQ?e=se8JLj) |
    | ScanNet200 | [B150/N50](../tools/cfgs/scannet200_models/spconv_clip_base150_caption.yaml) | 14.6 / 22.4 / 10.8 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EWdyFjCzvn9Gi0FN7VQAJl8BKcQAQYs61Lbh2V27ZT6h0g?e=FzOpex) |

- Annotation-free setup
  
    | Dataset | Model | mIoU  (mAcc) | Path |
    |:---:|:---:|:---:|:---:|
    | ScanNet | [RegionPLC + SparseUNet16](../tools/cfgs/scannet_models/zs/spconv_clip_caption_sparseunet16.yaml) | 56.9 (75.6) | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EWqYW58Q0GlKuhcAmWZYakUBYV0wyWfbxSarMHo0EZLfMg?e=1Ipfka) |
    | ScanNet | [RegionPLC + SparseUNet32](../tools/cfgs/scannet_models/zs/spconv_clip_caption_sparseunet32.yaml) | 59.6 (77.5) | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EdeusrJ9OnROsFYLd3E1vQMBdfydH5z5L7674K8tG4gYwQ?e=BTWn6j) |
    | ScanNet | [RegionPLC + OpenScene + SparseUNet16](../tools/cfgs/scannet_models/zs/spconv_clip_caption_sparseunet16_openscene.yaml) | 60.1 (74.4) | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EVdEJluknNdJhJ4H-oj8nrIBVIXBhZ7Wjw1m5nU68eM3AA?e=Y2V1pk) |
    | ScanNet | [RegionPLC + OpenScene + SparseUNet32](../tools/cfgs/scannet_models/zs/spconv_clip_caption_sparseunet32_openscene.yaml) | 63.6 (80.3)  | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EeleQjMKupFHnRDliBfCidIBLhC3xyewUJH4BSQTuh55HQ?e=ytbvVd) |
    | ScanNet200 | [RegionPLC + SparseUNet32](../tools/cfgs/scannet200_models/zs/spconv_clip_caption.yaml) |  9.1 (17.3) | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EdeusrJ9OnROsFYLd3E1vQMBdfydH5z5L7674K8tG4gYwQ?e=BTWn6j) |
    | ScanNet200 | [RegionPLC + OpenScene + SparseUNet32](../tools/cfgs/scannet200_models/zs/spconv_clip_caption_openscene.yaml) |  9.6 (17.8) | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EeleQjMKupFHnRDliBfCidIBLhC3xyewUJH4BSQTuh55HQ?e=ytbvVd) |


#### 3D Instance Segmentation
- Base-annotated setup

    | Dataset | Partition | hAP<sub>50</sub> / mAP<sub>50</sub>(B) / mAP<sub>50</sub>(N) | Path |
    |:---:|:---:|:---:|:---:|
    | ScanNet | [B13/N4](../tools/cfgs/scannet_models/inst/softgroup_clip_base13_caption.yaml) | 58.2 / 59.2 / 57.2 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/ETXofs-cwntBiitiBssSO_kBCKvmazufDxST2p9X7Mo56Q?e=zt8vSf) |
    | ScanNet | [B10/N7](../tools/cfgs/scannet_models/inst/softgroup_clip_base10_caption.yaml) | 40.6 / 53.9 / 32.5 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/Eak_DMQR07xCkB9g52Wqbn0B-j_VNHTQppom_r5K4CCvxQ?e=GpfG9T) |
    | ScanNet | [B8/N9](../tools/cfgs/scannet_models/inst/softgroup_clip_base8_caption.yaml) | 46.8 / 62.5 / 37.4 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EaNGtC64C5RMugxp8zISDK0BTOl7f7UZIcTR_lh1bsIEAQ?e=gbfaXz) |
