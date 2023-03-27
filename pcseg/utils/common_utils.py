import logging
import os
import io
import pickle
import random
import shutil
import subprocess
import SharedArray

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pathlib import Path


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, seed=666):
    if seed is not None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        torch.cuda.manual_seed(seed + worker_id)
        torch.cuda.manual_seed_all(seed + worker_id)


def get_pad_params(desired_size, cur_size):
    """
    Get padding parameters for np.pad function
    Args:
        desired_size: int, Desired padded output size
        cur_size: int, Current size. Should always be less than or equal to cur_size
    Returns:
        pad_params: tuple(int), Number of values padded to the edges (before, after)
    """
    assert desired_size >= cur_size

    # Calculate amount to pad
    diff = desired_size - cur_size
    pad_params = (0, diff)

    return pad_params


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method('spawn')
    # os.environ['MASTER_PORT'] = str(tcp_port)
    # os.environ['MASTER_ADDR'] = 'localhost'
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:%d' % tcp_port,
        rank=local_rank,
        world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank


def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    ndim = indices.shape[-1]
    flattened_indices = indices.view(-1, ndim)
    slices = [flattened_indices[:, i] for i in range(ndim)]
    ret[slices] = point_inds
    return ret


def generate_voxel2pinds(sparse_tensor):
    device = sparse_tensor.indices.device
    batch_size = sparse_tensor.batch_size
    spatial_shape = sparse_tensor.spatial_shape
    indices = sparse_tensor.indices.long()
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32)
    output_shape = [batch_size] + list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor


def sa_create(name, var):
    x = SharedArray.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnionGPU(output, target, K, ignore_index=-100):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1).clone()
    target = target.view(-1).clone()
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda(), area_output.cuda()


def update_meter(intersection_meter, union_meter, target_meter, output_meter, preds, labels,
                 n_classes, ignore_label=-100):
    intersection, union, target, output = intersectionAndUnionGPU(
        preds, labels, n_classes, ignore_label
    )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target), dist.all_reduce(output)

    intersection, union, target, output = \
        intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), output.cpu().numpy()

    intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
    output_meter.update(output)
    # precision = sum(intersection_meter.val) / (sum(output_meter.val) + 1e-10)
    accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
    return intersection_meter, union_meter, target_meter, output_meter, accuracy


def calc_metrics(intersection_meter, union_meter, target_meter, output_meter):
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    precision_class = intersection_meter.sum / (output_meter.sum + 1e-10)
    acc_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    label_ratio_class = output_meter.sum / (sum(output_meter.sum) + 1e-10)
    mIoU = np.mean(iou_class)
    mPre = np.mean(precision_class)
    mAcc = np.mean(acc_class)
    allPre = sum(intersection_meter.sum) / (sum(output_meter.sum) + 1e-10)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    return mIoU, mPre, mAcc, allPre, allAcc, iou_class, precision_class, acc_class, label_ratio_class


def update_binary_acc_meter(intersection_meter, target_meter, binary_preds, labels, idx1, n_classes):
    # idx1: binary_label: 0
    intersection, target = torch.zeros(n_classes).cuda(), torch.zeros(n_classes).cuda()
    binary_idx = [0 if i in idx1 else 1 for i in range(n_classes)]
    binary_idx = np.array(binary_idx)
    for c in range(n_classes):
        intersection[c] = (binary_preds[..., 0][labels == c] == binary_idx[c]).sum()
        target[c] = (labels == c).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(intersection), dist.all_reduce(target)
    intersection, target = intersection.cpu().numpy(), target.cpu().numpy()
    intersection_meter.update(intersection), target_meter.update(target)
    return intersection_meter, target_meter


def calc_binary_acc(intersection_meter, target_meter):
    acc_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mAcc = np.mean(acc_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    return mAcc, allAcc, acc_class


def merge_4_parts(x):
    """
    Helper function for S3IDS: take output of 4 parts and merge them.
    """
    inds = torch.arange(x.size(0), device=x.device)
    p1 = inds[::4]
    p2 = inds[1::4]
    p3 = inds[2::4]
    p4 = inds[3::4]
    ps = [p1, p2, p3, p4]
    x_split = torch.split(x, [p.size(0) for p in ps])
    x_new = torch.zeros_like(x)
    for i, p in enumerate(ps):
        x_new[p] = x_split[i]
    return x_new


def check_exists(path):
    if oss_data_client is not None:
        return oss_data_client.exist(path)
    elif isinstance(path, str):
        return os.path.exists(path)
    elif isinstance(path, Path):
        return path.exists()
    else:
        raise TypeError('Unexpected type for path: {}'.format(type(path)))


def sa_create(name, var):
    try:
        x = SharedArray.create(name, var.shape, dtype=var.dtype)
    except FileExistsError:
        return
    x[...] = var[...]
    x.flags.writeable = False
    return x


def sa_delete(name):
    try:
        SharedArray.delete(name)
    except:
        return


class OSSClient(object):
    def __init__(self, config_file='~/petreloss.conf'):
        from petrel_client.client import Client
        self.oss_client = Client(config_file)

    def get(self, file_path):
        file_bytes = self.oss_client.get(file_path)
        if file_bytes is None:
            raise ValueError(f'Unexpected path: {file_path}')

        # check extension type
        extension = os.path.splitext(file_path)[-1]
        if extension in ['.txt']:
            return file_bytes
        elif extension in ['.pkl', '.npy']:
            file_bytes = io.BytesIO(file_bytes)
        else:
            file_bytes = memoryview(file_bytes)

        return file_bytes

    def exist(self, file_path):
        file_bytes = self.oss_client.get(file_path)
        return file_bytes is not None

    def get_text(self, file_path):
        file_bytes = self.get(file_path)
        return str(file_bytes, encoding='utf-8').split('\n')


oss_data_client = None
