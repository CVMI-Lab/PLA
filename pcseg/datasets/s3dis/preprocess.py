# https://github.com/charlesq34/pointnet/blob/master/sem_seg/

import os
import sys
import glob
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

DATA_PATH = os.path.join('data', 's3dis/Stanford3dDataset_v1.2_Aligned_Version')
g_classes = [x.rstrip() for x in open(os.path.join(BASE_DIR, 'meta/class_names.txt'))]
g_class2label = {cls: i for i, cls in enumerate(g_classes)}
g_class2color = {'ceiling':	[0,255,0],
                 'floor':	[0,0,255],
                 'wall':	[0,255,255],
                 'beam':        [255,255,0],
                 'column':      [255,0,255],
                 'window':      [100,100,255],
                 'door':        [200,200,100],
                 'table':       [170,120,200],
                 'chair':       [255,0,0],
                 'sofa':        [200,100,100],
                 'bookcase':    [10,200,100],
                 'board':       [200,200,200],
                 'clutter':     [50,50,50]}
g_easy_view_labels = [7,8,9,10,11,1]
g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}


def collect_point_label(anno_path, out_filename, file_format='txt'):
    """ Convert original dataset files to data_label file (each line is XYZRGBL).
        We aggregated all the points from each instance in the room.
    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save collected points and labels (each line is XYZRGBL)
        file_format: txt or numpy, determines what file format to save.
    Returns:
        None
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """
    points_list = []

    num_inst = 0
    for f in sorted(glob.glob(os.path.join(anno_path, '*.txt'))):
        cls = os.path.basename(f).split('_')[0]
        num_inst += 1
        if cls not in g_classes: # note: in some room there is 'staris' class..
            cls = 'clutter'
        points = np.loadtxt(f)
        labels = np.ones((points.shape[0], 1)) * g_class2label[cls]
        inst_labels = np.ones((points.shape[0], 1)) * num_inst
        points_list.append(np.concatenate([points, labels, inst_labels], 1)) # Nx8

    data_label = np.concatenate(points_list, 0)
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min

    if file_format == 'txt':
        fout = open(out_filename, 'w')
        for i in range(data_label.shape[0]):
            fout.write('%f %f %f %d %d %d %d %d\n' % \
                       (data_label[i, 0], data_label[i, 1], data_label[i, 2],
                        data_label[i, 3], data_label[i, 4], data_label[i, 5],
                        data_label[i, 6], data_label[i, 7]))
        fout.close()
    elif file_format == 'numpy':
        np.save(out_filename, data_label)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
              (file_format))
        exit()


def main():
    anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
    anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

    output_folder = './data/s3dis/stanford_indoor3d_inst'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
    for anno_path in anno_paths:
        print(anno_path)
        # try:
        elements = anno_path.split('/')
        out_filename = elements[-3]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy
        collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
        # except:
        #     print(anno_path, 'ERROR!!')


if __name__ == '__main__':
    main()
