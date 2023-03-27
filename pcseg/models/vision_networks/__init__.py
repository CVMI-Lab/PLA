from .network_template import ModelTemplate
from .sparseunet_textseg import SparseUNetTextSeg

__all__ = {
    'ModelTemplate': ModelTemplate,
    'SparseUNetTextSeg': SparseUNetTextSeg
}


def build_model(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
