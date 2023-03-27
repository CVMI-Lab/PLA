from .text_seg_head import TextSegHead
from .binary_head import BinaryHead
from .caption_head import CaptionHead
from .linear_head import LinearHead
from .inst_head import InstHead

__all__ = {
    'TextSegHead': TextSegHead,
    'BinaryHead': BinaryHead,
    'CaptionHead': CaptionHead,
    'LinearHead': LinearHead,
    'InstHead': InstHead
}
