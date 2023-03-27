import os
import glob


def load_best_metric(ckpt_save_dir):
    best_metric, best_epoch = 0.0, -1
    best_metric_record_list = glob.glob(str(ckpt_save_dir / '*.txt'))
    if len(best_metric_record_list) > 0:
        best_metric_record_name = os.path.basename(best_metric_record_list[0])
        best_split_list = os.path.splitext(best_metric_record_name)[0].split('_')
        best_metric = float(best_split_list[2])
        best_epoch = int(best_split_list[-1])
    return best_metric, best_epoch
