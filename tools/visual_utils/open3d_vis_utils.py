"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

COLOR20 = np.array(
        [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
        [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
        [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])

COLOR40 = np.array(
        [[88,170,108], [174,105,226], [78,194,83], [198,62,165], [133,188,52], [97,101,219], [190,177,52], [139,65,168],[75,202,137], [225,66,129],
        [68,135,42], [226,116,210], [146,186,98], [68,105,201], [219,148,53], [85,142,235], [212,85,42], [78,176,223], [221,63,77], [68,195,195],
        [175,58,119], [81,175,144], [184,70,74], [40,116,79], [184,134,219], [130,137,46], [110,89,164], [92,135,74], [220,140,190], [94,103,39],
        [144,154,219], [160,86,40], [67,107,165], [194,170,104], [162,95,150], [143,110,44], [146,72,105], [225,142,106], [162,83,86], [227,124,143]])


# scannet
SCANNET_SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SCANNET_SEMANTIC_NAMES = np.array(['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                                   'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator',
                                   'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'])
SCANNET_DA_SEMANTIC_NAMES = np.array(["wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window",
                                      "bookshelf", "desk", "ceiling", "unannotated"])
SCANNET_CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'floor': [143, 223, 142],
    'wall': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [80, 83, 160],
    'ceiling': [0, 255, 0]
}
SCANNET_SEMANTIC_IDX2NAME = {1: 'wall', 2: 'floor', 3: 'cabinet', 4: 'bed', 5: 'chair', 6: 'sofa', 7: 'table',
                             8: 'door', 9: 'window', 10: 'bookshelf', 11: 'picture', 12: 'counter', 14: 'desk',
                             16: 'curtain', 24: 'refridgerator', 28: 'shower curtain', 33: 'toilet',  34: 'sink',
                             36: 'bathtub', 39: 'otherfurniture'}


# s3dis
S3DIS_SEMANTIC_NAMES = np.array(["ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair",
                                 "sofa", "bookshelf", "board", "clutter"])
S3DIS_DA_SEMANTIC_NAMES = np.array(["wall", "floor", "chair", "sofa", "table", "door", "window", "bookshelf",
                                    "ceiling", "beam", "column", 'ignore'])

S3DIS_SEMANTIC_IDX2NAME = {1: 'ceiling', 2: 'floor', 3: 'wall', 4: 'beam', 5: 'column', 6: 'window', 7: 'door',
                           8: 'table', 9: 'chair', 10: 'sofa', 11:'bookshelf', 12: 'board', 13: 'clutter'}

S3DIS_CLASS_COLOR = {
    'ceiling': [0, 255, 0],
    'floor': [0, 0, 255],
    'wall': [0, 255, 255],
    'beam': [255, 255, 0],
    'column': [255, 0, 255],
    'window': [100, 100, 255],
    'door': [200, 200, 100],
    'table': [170, 120, 200],
    'chair': [255, 0, 0],
    'sofa': [200, 100, 100],
    'bookshelf': [10, 200, 100],
    'board': [200, 200, 200],
    'clutter': [50, 50, 50],
    'ignore': [0, 0, 0]
}

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None,
                point_colors=None, draw_origin=True, point_size=1.0):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    # import ipdb; ipdb.set_trace(context=20)
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = point_size
    # vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        color = np.array([0.44, 0.514, 0.655])
        pts.colors = open3d.utility.Vector3dVector(np.repeat(color[None, :], points.shape[0], axis=0))
    else:
        # normalize color to [0, 1]
        point_colors = (point_colors - point_colors.min()) / (point_colors.max() - point_colors.min())
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def draw_scenes_v2(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None,
                   point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    app = open3d.visualization.gui.Application.instance
    app.initialize()
    vis = open3d.visualization.O3DVisualizer()
    vis.show_settings = True

    vis.point_size = 2
    vis.set_background(np.zeros(4), None)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry('origin', axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry('points', pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3), dtype=np.float32))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0.0, 0.0, 1.0), name='gt_box')

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0.0, 1.0, 0.0), ref_labels, ref_scores, name='pred_box')

    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()


def draw_seg_results(points, seg_labels=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if seg_labels is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        point_colors = get_coor_colors(seg_labels)
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis


def dump_vis_dict(vis_dict, path='./vis_dict.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(vis_dict, f)


def plot_image_with_caption(image, caption, image_name=None):
    plt.imshow(image)
    plt.xlabel(caption)
    if image_name is not None:
        plt.savefig('vis_output/' + image_name + '.png')
    else:
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--path', type=str, default='./vis_dict.pkl', help='specify the data of dataset')

    args = parser.parse_args()

    with open(args.path, 'rb') as f:
        vis_dict = pickle.load(f)

    draw_scenes(**vis_dict)

