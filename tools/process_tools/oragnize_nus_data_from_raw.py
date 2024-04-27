"""
Copyright: Jihan YANG
from 2022 - present
"""
import os
import subprocess


meta_data_name_list = ['v1.0-trainval_meta.tgz']
main_data_name_list = ['v1.0-trainval01_blobs_lidar.tar', 'v1.0-trainval02_blobs_lidar.tar',
                       'v1.0-trainval03_blobs_lidar.tar', 'v1.0-trainval04_blobs_lidar.tar',
                       'v1.0-trainval05_blobs_lidar.tar', 'v1.0-trainval06_blobs_lidar.tar',
                       'v1.0-trainval07_blobs_lidar.tar', 'v1.0-trainval08_blobs_lidar.tar',
                       'v1.0-trainval09_blobs_lidar.tar', 'v1.0-trainval10_blobs_lidar.tar',]
lidar_seg_label_name_list = ['nuScenes-lidarseg-all-v1.0.tar']


def main(args):
    # mkdir v1.0-trainval
    os.makedirs(os.join('./', args.tag), exist_ok=True)

    # $ cd v1.0-trainval
    os.chdir(args.data)

    # extract meta files
    for name in meta_data_name_list:
        subprocess.run('tar', '-zxvf', name)

    # extract main files
    for name in main_data_name_list:
        subprocess.run('tar', '-xvf', name)

    # extract seg label files
    for name in lidar_seg_label_name_list:
        subprocess.run('tar', '-xvf', name)


if __name__ == '__main__':
    """
    make sure your path is under nuscenes
    """
    import argparse

    parser = argparse.ArgumentParser('language model')
    parser.add_argument('--tag', default='v1.0-trainval', type=str, help='language model name')
    args = parser.parse_args()
