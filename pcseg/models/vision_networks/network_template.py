import os

import torch
import torch.nn as nn

from ...utils.spconv_utils import find_all_spconv_keys
from ..vision_backbones_3d import vfe
from .. import vision_backbones_3d, head, adapter
from pcseg.config import cfg


class ModelTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.fixed_modules = model_cfg.get('FIXED_MODULES', [])
        # For S3DIS
        self.test_x4_split = dataset.dataset_cfg.DATA_PROCESSOR.get('x4_split', False)

        self.module_topology = [
            'vfe', 'backbone_3d', 'adapter', 'binary_head', 'kd_head', 'task_head', 'inst_head', 'caption_head'
        ]
        self.module_list = self.build_networks()

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=self.dataset.dataset_cfg.get('NUM_POINT_FEATURES', None),
            voxel_size=self.dataset.dataset_cfg.get('VOXEL_SIZE', None),
            voxel_mode=self.dataset.dataset_cfg.DATA_PROCESSOR.get('voxel_mode', None),
        )
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone3d_module = vision_backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D
        )
        model_info_dict['backbone3d_out_channel'] = self.model_cfg.BACKBONE_3D.MID_CHANNEL
        model_info_dict['module_list'].append(backbone3d_module)
        return backbone3d_module, model_info_dict

    def build_task_head(self, model_info_dict):
        if self.model_cfg.get('TASK_HEAD', None) is None:
            return None, model_info_dict
        if 'IN_CHANNEL' in self.model_cfg.TASK_HEAD:
            in_channel = self.model_cfg.TASK_HEAD.IN_CHANNEL
        else:
            in_channel = model_info_dict['backbone3d_out_channel']
        task_head_module = head.__all__[self.model_cfg.TASK_HEAD.NAME](
            model_cfg=self.model_cfg.TASK_HEAD,
            in_channel=in_channel,
            ignore_label=self.dataset.ignore_label,
            num_class=self.num_class
        )
        model_info_dict['module_list'].append(task_head_module)
        return task_head_module, model_info_dict

    def build_inst_head(self, model_info_dict):
        if self.model_cfg.get('INST_HEAD', None) is None:
            return None, model_info_dict

        if 'IN_CHANNEL' in self.model_cfg.TASK_HEAD:
            in_channel = self.model_cfg.TASK_HEAD.IN_CHANNEL
        else:
            in_channel = model_info_dict['backbone3d_out_channel']

        if hasattr(self.dataset, 'base_inst_class_idx'):
            base_inst_class_idx = self.dataset.base_inst_class_idx
            novel_inst_class_idx = self.dataset.novel_inst_class_idx
        else:
            base_inst_class_idx = novel_inst_class_idx = None

        inst_head_module = head.__all__[self.model_cfg.INST_HEAD.NAME](
            model_cfg=self.model_cfg.INST_HEAD,
            in_channel=in_channel,
            inst_class_idx=self.dataset.inst_class_idx,
            sem2ins_classes=self.dataset.dataset_cfg.sem2ins_classes,
            valid_class_idx=self.dataset.valid_class_idx,
            label_shift=self.dataset.inst_label_shift,
            ignore_label=self.dataset.ignore_label,
            base_inst_class_idx=base_inst_class_idx,
            novel_inst_class_idx=novel_inst_class_idx
        )
        model_info_dict['module_list'].append(inst_head_module)
        return inst_head_module, model_info_dict

    def build_adapter(self, model_info_dict):
        if self.model_cfg.get('ADAPTER', None) is None:
            return None, model_info_dict

        adapter_module = adapter.__all__[self.model_cfg.ADAPTER.NAME](
            model_cfg=self.model_cfg.ADAPTER,
            in_channel=model_info_dict['backbone3d_out_channel'],
        )
        model_info_dict['module_list'].append(adapter_module)
        return adapter_module, model_info_dict

    def build_binary_head(self, model_info_dict):
        if self.model_cfg.get('BINARY_HEAD', None) is None:
            return None, model_info_dict

        binary_head_module = head.__all__[self.model_cfg.BINARY_HEAD.NAME](
            model_cfg=self.model_cfg.BINARY_HEAD,
            ignore_label=self.dataset.ignore_label,
            in_channel=model_info_dict['backbone3d_out_channel'],
            block_reps=self.model_cfg.BACKBONE_3D.BLOCK_REPS,
            block_residual=self.model_cfg.BACKBONE_3D.BLOCK_RESIDUAL
        )
        model_info_dict['module_list'].append(binary_head_module)
        return binary_head_module, model_info_dict

    def build_caption_head(self, model_info_dict):
        if self.model_cfg.get('CAPTION_HEAD', None) is None:
            return None, model_info_dict

        caption_head_module = head.__all__[self.model_cfg.CAPTION_HEAD.NAME](
            model_cfg=self.model_cfg.CAPTION_HEAD,
            ignore_label=self.dataset.ignore_label
        )
        model_info_dict['module_list'].append(caption_head_module)
        return caption_head_module, model_info_dict

    def build_kd_head(self, model_info_dict):
        if self.model_cfg.get('KD_HEAD', None) is None:
            return None, model_info_dict

        kd_head_module = head.__all__[self.model_cfg.KD_HEAD.NAME](
            model_cfg=self.model_cfg.KD_HEAD
        )
        model_info_dict['module_list'].append(kd_head_module)
        return kd_head_module, model_info_dict

    def forward(self, batch_dict):
        batch_dict['test_x4_split'] = self.test_x4_split
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        ret_dict = self.task_head.forward_ret_dict
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict['loss'] = loss
            return ret_dict, tb_dict, disp_dict
        else:
            return ret_dict

    def get_training_loss(self):
        disp_dict = {}
        # for segmentation loss
        loss, tb_dict = self.task_head.get_loss()

        tb_dict['loss'] = loss.item()
        disp_dict.update(tb_dict)

        return loss, tb_dict, disp_dict

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, epoch_id=None, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        try:
            model_state_disk = checkpoint['model_state']
        except:
            model_state_disk = checkpoint['state_dict']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        if self.model_cfg.get('REMAP_FROM_3DLANG', None):
            model_state_disk = self.remap_keys_from_3dlang(model_state_disk)
        elif self.model_cfg.get('REMAP_FROM_LAI', None):
            model_state_disk = self.remap_keys_from_lai(model_state_disk)
        elif self.model_cfg.get('REMAP_FROM_NOADAPTER', None):
            model_state_disk = self.remap_keys_from_noadapter(model_state_disk)
        elif self.model_cfg.get('REMAP_FROM_KDADAPTER', None):
            model_state_disk = self.remap_keys_from_kdadapter(model_state_disk)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

        if epoch_id and epoch_id == 'no_number' and 'epoch' in checkpoint:
            epoch_id = checkpoint['epoch']
        return epoch_id

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch

    @staticmethod
    def remap_keys_from_3dlang(model_state_disk):
        """
        To remap key from 3d-lang into this repo
        Args:
            model_state_disk:

        Returns:

        """
        new_model_state = {}
        for key in model_state_disk.keys():
            first_name = key.split('.')[0]
            if first_name in ['input_conv', 'unet', 'output_layer']:
                new_key = 'backbone_3d.' + key
            elif first_name in ['semantic_linear']:
                new_key = 'task_head.cls_head' + key[len(first_name):]
            elif first_name == 'blinear':
                new_key = 'binary_head.binary_encoder' + key[len(first_name):]
            elif first_name == 'blinear_outputlayer':
                new_key = 'binary_head.binary_classifier' + key[len(first_name):]
            elif first_name == 'blinear_final':
                new_key = 'binary_head.binary_classifier.2' + key[len(first_name):]
            elif first_name == 'adapter':
                new_key = 'adapter.' + key
            elif first_name == 'tiny_unet' or first_name == 'tiny_unet_outputlayer' or \
                first_name == 'mask_linear' or first_name == 'iou_score_linear' or first_name == 'offset_linear':
                new_key = 'inst_head.' + key
            elif first_name == 'caption_logit_scale':
                new_key = 'caption_head.' + key
            else:
                continue
            new_model_state[new_key] = model_state_disk[key]

        return new_model_state

    @staticmethod
    def remap_keys_from_lai(model_state_disk):
        """
        To remap key from 3d-lang into this repo
        Args:
            model_state_disk:

        Returns:

        """
        new_model_state = {}
        for key in model_state_disk.keys():
            name_list = key.split('.')
            first_name = name_list[0]
            if first_name in ['module']:
                if name_list[1] == 'linear':
                    new_key = 'task_head.cls_head' + key[len(first_name)+len(name_list[1])+1:]
                else:
                    new_key = 'backbone_3d' + key[len(first_name):]
            else:
                continue

            new_model_state[new_key] = model_state_disk[key]

        return new_model_state

    def remap_keys_from_noadapter(self, model_state_disk):
        """
        To remap key from no adapter version into this repo
        Args:
            model_state_disk:

        Returns:

        """
        new_model_state = {}
        for key in model_state_disk.keys():
            name_list = key.split('.')
            first_name = name_list[0]
            new_key = key
            if first_name in ['task_head']:
                if name_list[1] == 'adapter':
                    new_key = 'adapter.adapter' + key[len(first_name)+len(name_list[1])+1:]

            new_model_state[new_key] = model_state_disk[key]

        return new_model_state

    def remap_keys_from_kdadapter(self, model_state_disk):
        """
        To remap key from kd adapter version into this repo
        Args:
            model_state_disk:

        Returns:

        """
        new_model_state = {}
        for key in model_state_disk.keys():
            name_list = key.split('.')
            first_name = name_list[0]
            new_key = key
            if first_name in ['kd_head']:
                if name_list[1] == 'kd_adapter':
                    new_key = 'adapter.adapter' + key[len(first_name)+len(name_list[1])+len(name_list[2])+2:]
                # print(key, new_key, model_state_disk[key].shape)
            new_model_state[new_key] = model_state_disk[key]

        return new_model_state
