The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), and the model configs are located within [tools/cfgs](../tools/cfgs) for different settings.

#### ScanNet Dataset
- Please download the [ScanNet Dataset](http://www.scan-net.org/) and follow [PointGroup](https://github.com/dvlab-research/PointGroup/blob/master/dataset/scannetv2/prepare_data_inst.py) to pre-process the dataset as follows or directly download the pre-processed data [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3007346_connect_hku_hk/EpTBva1Ev0BLu7TYz_03UUQBpLnyFlijK9z645tavor68w?e=liM2HD).
- Additionally, please download the caption data [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3007346_connect_hku_hk/EpTBva1Ev0BLu7TYz_03UUQBpLnyFlijK9z645tavor68w?e=liM2HD). If you want to generate captions on your own, please download image data ([scannet_frames_25k]((http://www.scan-net.org/))) from ScanNet and follow scripts [generate_caption.py](../tools/process_tools/generate_caption.py) and [generate_caption_idx.py](../tools/process_tools/generate_caption_idx.py).

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

#### S3DIS Dataset
- Please download the [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html#Download) and follow [dataset/s3dis/preprocess.py](../dataset/s3dis/preprocess.py) to pre-process the dataset as follows or directly download the pre-processed data [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3007346_connect_hku_hk/EoNAsU5f8YRGtQYV8ewhwvQB7QPbxT-uwKqTk8FPiyUTtQ?e=wq58H7).
    ```bash
    python3 pcseg/datasets/s3dis/preprocess.py 
    ```
    
- Additionally, please download the caption data [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3007346_connect_hku_hk/EoNAsU5f8YRGtQYV8ewhwvQB7QPbxT-uwKqTk8FPiyUTtQ?e=wq58H7). If you want to generate captions on your own, please download image data [here](https://github.com/alexsax/2D-3D-Semantics) and follows scripts here: [generate_caption.py](../tools/process_tools/generate_caption.py) and [generate_caption_idx.py](../tools/process_tools/generate_caption_idx.py).
 
- The directory organization should be as follows:

    ```
    PLA
    ├── data
    │   ├── s3dis
    │   │   │── stanford_indoor3d_inst
    │   │   │   │── Area_1_Conference_1.npy
    │   │   │   │── ...
    │   │   │── text_embed
    │   │   │── caption_idx
    │   │   │—— s3dis_2d (optional, only for caption generation)
    ├── pcseg
    ├── tools
    ```
