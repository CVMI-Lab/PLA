from plyfile import PlyData
import numpy as np


def read_ply(path):
    plydata = PlyData.read(path)
    num_verts = plydata['vertex'].count

    vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
    vertices[:, 0] = plydata['vertex']['x']
    vertices[:, 1] = plydata['vertex']['y']
    vertices[:, 2] = plydata['vertex']['z']

    rgb = np.zeros(shape=[num_verts, 3], dtype=np.float32)
    rgb[:, 0] = plydata['vertex']['red']
    rgb[:, 1] = plydata['vertex']['green']
    rgb[:, 2] = plydata['vertex']['blue']
    alpha = np.array(plydata['vertex']['alpha'])

    face_indices = plydata['face']['vertex_indices']

    return vertices, rgb, alpha, face_indices


def write_ply(output_file, data_dict):
    verts, colors = data_dict['xyz'], data_dict['rgb']
    if 'indices' not in data_dict:
        data_dict['indices'] = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    if 'alpha' in data_dict:
        file.write('property uchar alpha\n')
    file.write('element face {:d}\n'.format(len(data_dict['indices'])))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')

    if 'alpha' in data_dict:
        for vert, color, a in zip(verts, colors, data_dict['alpha']):
            file.write('{:f} {:f} {:f} {:d} {:d} {:d} {:d}\n'.format(vert[0], vert[1], vert[2],
                                                                     int(color[0]),
                                                                     int(color[1]),
                                                                     int(color[2]),
                                                                     int(a)))
    else:
        for vert, color in zip(verts, colors):
            file.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(vert[0], vert[1], vert[2],
                                                                int(color[0]),
                                                                int(color[1]),
                                                                int(color[2])))
    for ind in data_dict['indices']:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()
