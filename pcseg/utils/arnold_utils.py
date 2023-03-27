import os


class ArnoldUtils():
    def __init__(self, enabled, arnold_dir, logger) -> None:
        self.enabled = enabled
        self.logger = logger
        self.dir = arnold_dir

    def save_ckpt(self, ckpt_path, last_epoch=False):
        if self.enabled:
            ckpt_dir, file_name = os.path.split(ckpt_path)
            # import ipdb; ipdb.set_trace(context=10)
            _ckpt_dir = ckpt_dir[ckpt_dir.find('output'):][7:]
            os.system('hdfs dfs -mkdir -p hdfs://haruna/home/byte_arnold_hl_vc/user/ryding/{}/{}'.format(self.dir, _ckpt_dir))
            if last_epoch:
                tgt_path = os.path.join(self.dir, _ckpt_dir, 'last_train.pth')
            else:
                tgt_path = os.path.join(self.dir, _ckpt_dir, file_name)
            os.system('hdfs dfs -put -f {} hdfs://haruna/home/byte_arnold_hl_vc/user/ryding/{}'.format(ckpt_path, tgt_path))
            self.logger.info('Put model to hdfs://haruna/home/byte_arnold_hl_vc/user/ryding/{}'.format(tgt_path))

    def load_ckpt(self, ckpt_dir):
        if self.enabled:
            try:
                _ckpt_dir = ckpt_dir[ckpt_dir.find('output'):][7:]
                os.system('hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/user/ryding/{}/{}/* {}'.format(self.dir, _ckpt_dir, ckpt_dir))
                self.logger.info('Get model from hdfs://haruna/home/byte_arnold_hl_vc/user/ryding/{}/{}'.format(self.dir,_ckpt_dir))
            except:
                pass
