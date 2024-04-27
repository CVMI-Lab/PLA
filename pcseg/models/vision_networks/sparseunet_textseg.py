import torch
from .network_template import ModelTemplate


class SparseUNetTextSeg(ModelTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg, num_class, dataset)
        if model_cfg.get('BINARY_HEAD', None):
            self.binary_head.register_hook_for_binary_head(self.backbone_3d)

    def forward(self, batch_dict):
        batch_dict['test_x4_split'] = self.test_x4_split
        # Order: vfe, backbone_3d, binary_head, seg_head, caption_head
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        ret_dict = self.task_head.forward_ret_dict
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict['loss'] = loss
            return ret_dict, tb_dict, disp_dict
        else:
            if hasattr(self, 'inst_head') and self.inst_head is not None:
                ret_dict.update(self.inst_head.forward_ret_dict)
            return ret_dict

    def get_training_loss(self):
        disp_dict = {}
        tb_dict = {}

        # for segmentation loss
        if not self.task_head.eval_only:
            seg_loss, tb_dict_seg = self.task_head.get_loss()
            tb_dict.update(tb_dict_seg)
        else:
            seg_loss = 0

        # for binary loss
        if self.binary_head is not None:
            binary_loss, tb_dict_binary = self.binary_head.get_loss()
            tb_dict.update(tb_dict_binary)
        else:
            binary_loss = 0

        # for caption loss
        if self.caption_head is not None:
            caption_loss, tb_dict_caption = self.caption_head.get_loss()
            tb_dict.update(tb_dict_caption)
        else:
            caption_loss = 0

        # for inst loss
        if self.inst_head is not None:
            inst_loss, tb_dict_inst = self.inst_head.get_loss()
            tb_dict.update(tb_dict_inst)
        else:
            inst_loss = 0

        # for distillation loss
        if self.kd_head is not None:
            kd_loss, tb_dict_kd = self.kd_head.get_loss()
            tb_dict.update(tb_dict_kd)
        else:
            kd_loss = 0

        loss = seg_loss + binary_loss + caption_loss + inst_loss + kd_loss
        tb_dict['loss'] = loss.item()
        disp_dict.update(tb_dict)

        return loss, tb_dict, disp_dict
