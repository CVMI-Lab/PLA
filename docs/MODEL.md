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
    | ScanNet | B15/N4 | 64.9 / 67.8 / 62.2 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/Ef8xk_X0ortMjC0F8PBQl2wBacVPgO72La8h_ZTDsKj__Q?e=Uq6W8I) |
    | ScanNet | B12/N7 | 55.9 / 70.4 / 46.4 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/EVl7SdeUEPFAvrj2xnWSb-sBCOtWYyVOwBo6ggFb9x7dNA?e=feZaxH) |
    | ScanNet | B10/N9 | 52.8 / 76.6 / 40.3 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/Ef0P_6XraDpCo0RRgOJ1wGQB-xOW7T6lecvVRi5P90Edbw?e=hqrP8X) |
    | S3DIS | B8/N4 |  35.6 / 58.3 / 25.6 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/EYIW4SNX5B9Go_LKiim1KFEB_abYv0bDZMggE_6Ifjau0g?e=8BD0K3) |
    | S3DIS | B6/N6 | 38.4 / 53.9 / 29.8 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/EeNYtkS3pmhAvc3Hxj7__SwB8SMzZdzmljRtCYuYG8NHcA?e=aC0aE2) |


- instance segmentation

    | Dataset | Partition | hAP<sub>50</sub> / mAP<sub>50</sub>(B) / mAP<sub>50</sub>(N) | Path |
    |:---:|:---:|:---:|:---:|
    | ScanNet | B13/N4 | 57.8 / 58.7 / 56.9| [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/Eb4N2hfCevlBlBxWlK9DtioBP6RX7gtXUmY0Huu4MknUHA?e=YDydlj) |
    | ScanNet | B10/N7 | 31.6 / 54.8 / 22.2 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/ETsHZCFElvdCmk8ulRzBk-EBxm8fHk8rLJnpUdk9_n3i1Q?e=4SGy1N) |
    | ScanNet | B8/N9 | 36.9 / 63.1 / 26.2 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/EXAaU8RDecJFn_1J2Q-IqdsBALbv-5d_L_RyIOrdIjB66g?e=c8dFD6) |
    | S3DIS | B8/N4 | 17.2 / 60.9 / 10.0| [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/ETzzD-pEhvtMkJGnIxzgIP0Bk3f2He9_hkgfVtexEMFqpg?e=xJpaOV) |
    | S3DIS | B6/N6 |15.8 / 48.2 / 9.5| [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/EWoqIoBWfSRBqQwahLTKQGkB5Gwp8zs0EvT3MkGMDiBOrw?e=daBppj) |

