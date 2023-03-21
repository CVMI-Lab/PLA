The dataset configs are located within [cfgs/dataset_configs](../cfgs/dataset_configs), and the model configs are located within [cfgs](../cfgs) for different settings.

#### ScanNet Dataset
- Please download the [ScanNet Dataset](http://www.scan-net.org/) and follow [PointGroup](https://github.com/dvlab-research/PointGroup/blob/master/dataset/scannetv2/prepare_data_inst.py) to pre-process the dataset as follows or directly download the pre-processed data [here](). Additionally, please download the caption data [here]().

    ```
    PLA
    ├── data
    │   ├── scannetv2
    │   │   │── train
    │   │   │   │── scene0000_00.pth
    │   │   │   │── ...
    │   │   │── val
    │   │   │── text_embed
    │   │   │── scannetv2_matching_idx.pickle
    │   │   │── scannetv2_matching_idx_intersect_v3.pickle
    ├── cfgs
    ├── dataset
    ```

#### S3DIS Dataset
- Please download the [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html#Download) and follow [PointNet](https://github.com/charlesq34/pointnet/blob/master/sem_seg/collect_indoor3d_data.py) to pre-process the dataset as follows or directly download the pre-processed data [here](). Additionally, please download the caption data [here]().
    ```
    PLA
    ├── data
    │   ├── s3dis
    │   │   │── stanford_indoor3d_inst
    │   │   │   │── Area_1_Conference_1.npy
    │   │   │   │── ...
    │   │   │── text_embed
    │   │   │── s3dis_matching_idx
    │   │   │── s3dis_matching_idx_intersect_v3
    ├── cfgs
    ├── dataset
    ```
