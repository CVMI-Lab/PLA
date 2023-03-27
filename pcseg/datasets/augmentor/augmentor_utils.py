import math
import numpy as np
import scipy.ndimage
import scipy.interpolate
import scipy.stats


def check_key(key):
    exist = key is not None
    if not exist:
        return False
    if isinstance(key, bool):
        enabled = key
    elif isinstance(key, dict):
        enabled = key.get('enabled', True)
    else:
        enabled = True
    return enabled


def check_p(key):
    return (not isinstance(key, dict)) or ('p' not in key) or (np.random.rand() < key['p'])


def elastic(x, gran, mag):
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3

    bb = np.abs(x).max(0).astype(np.int32) // gran + 3
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

    def g(x_):
        return np.hstack([i(x_)[:, None] for i in interp])

    return x + g(x) * mag


def scene_aug(aug, xyz, rgb=None):
    assert xyz.ndim == 2
    m = np.eye(3)
    if check_key(aug.jitter):
        m += np.random.randn(3, 3) * 0.1
    if check_key(aug.flip) and check_p(aug.flip):
        m[0][0] *= -1  # np.random.randint(0, 2) * 2 - 1  # flip x randomly
    if check_key(aug.rotation) and check_p(aug.rotation):
        theta_x = (np.random.rand() * 2 * math.pi - math.pi) * aug.rotation.value[0]
        theta_y = (np.random.rand() * 2 * math.pi - math.pi) * aug.rotation.value[1]
        theta_z = (np.random.rand() * 2 * math.pi - math.pi) * aug.rotation.value[2]
        Rx = np.array \
            ([[1, 0, 0], [0, math.cos(theta_x), -math.sin(theta_x)], [0, math.sin(theta_x), math.cos(theta_x)]])
        Ry = np.array \
            ([[math.cos(theta_y), 0, math.sin(theta_y)], [0, 1, 0], [-math.sin(theta_y), 0, math.cos(theta_y)]])
        Rz = np.array \
            ([[math.cos(theta_z), math.sin(theta_z), 0], [-math.sin(theta_z), math.cos(theta_z), 0], [0, 0, 1]])
        rot_mats = [Rx, Ry, Rz]
        if aug.rotation.get('shuffle', False):
            np.random.shuffle(rot_mats)
        m = np.matmul(m, rot_mats[0].dot(rot_mats[1]).dot(rot_mats[2]))
    xyz = np.matmul(xyz, m)
    if check_key(aug.random_jitter) and check_p(aug.random_jitter):
        if aug.random_jitter.accord_to_size:
            jitter_scale = (xyz.max(0) - xyz.min(0)).mean() * 0.1
        else:
            jitter_scale = aug.random_jitter.value
        random_noise = (np.random.rand(xyz.shape[0], xyz.shape[1]) - 0.5) * jitter_scale
        xyz += random_noise
    if check_key(aug.scaling_scene) and check_p(aug.scaling_scene):
        scaling_fac = np.random.rand() * (aug.scaling_scene.value[1] - aug.scaling_scene.value[0]) \
                      + aug.scaling_scene.value[0]
        xyz_center = (xyz.max(0) + xyz.min(0)) / 2.0
        xyz = (xyz - xyz_center) * scaling_fac + xyz_center

    if rgb is not None and check_key(aug.color_jitter):
        rgb += np.random.randn(3) * 0.1
    return xyz, rgb


def crop(xyz, full_scale, max_npoint, step=32):
    xyz_offset = xyz.copy()
    valid_idxs = (xyz_offset.min(1) >= 0)
    assert valid_idxs.sum() == xyz.shape[0]
    full_scale = np.array([full_scale[1]] * 3)
    room_range = xyz.max(0) - xyz.min(0)

    while valid_idxs.sum() > max_npoint:
        step_temp = step
        if valid_idxs.sum() > 1e6:
            step_temp = step * 2
        offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
        xyz_offset = xyz + offset
        valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
        full_scale[:2] -= step_temp

    return xyz_offset, valid_idxs

