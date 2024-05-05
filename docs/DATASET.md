The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), and the model configs are located within [tools/cfgs](../tools/cfgs) for different settings.

#### ScanNet Dataset
- Please download the [ScanNet Dataset](http://www.scan-net.org/) and follow [PointGroup](https://github.com/dvlab-research/PointGroup/blob/master/dataset/scannetv2/prepare_data_inst.py) to pre-process the dataset as follows or directly download the pre-processed data [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3007346_connect_hku_hk/EpTBva1Ev0BLu7TYz_03UUQBpLnyFlijK9z645tavor68w?e=liM2HD).
- Additionally, please download the caption data [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jhyang13_connect_hku_hk/EsEgHtGOWbJImxop6tF15FIBIH-ihrjuDclbrbmwWfY_RA?e=s6fitN). If you want to generate captions on your own, please download image data ([scannet_frames_25k]((http://www.scan-net.org/))) from ScanNet and follow scripts [generate_caption.py](../tools/process_tools/generate_caption.py) and [generate_caption_idx.py](../tools/process_tools/generate_caption_idx.py).

- The directory organization should be as follows:

    ```
    PLA
    ├── data
    │   ├── scannetv2
    │   │   │── train
    │   │   │   │── scene0000_00.pth
    │   │   │   │── ...
    │   │   │── val
    │   │   │── text_embed
    │   │   │── caption_idx
    │   │   │── scannetv2_train.txt
    │   │   │── scannetv2_val.txt
    │   │   │—— scannet_frames_25k (optional, only for caption generation)
    ├── pcseg
    ├── tools
    ```


#### nuScenes Dataset
* Please download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and organize the downloaded files as follows: 
* Additionally, please download the caption data [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jhyang13_connect_hku_hk/Eh9qCTiV0VBEuJDLRai7_MUBGOepuHr3F9y-VVnIjhyELw?e=a4yXcB).
```
PLA
├── data
│   ├── nuscenes
│   │   │── text_embed
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── caption_idx
│   │   │   │── v1.0-trainval
├── pcseg
├── tools
```

* Install the `nuscenes-devkit` with version `1.0.5` by running the following command: 
```shell script
pip install nuscenes-devkit==1.0.5
```
