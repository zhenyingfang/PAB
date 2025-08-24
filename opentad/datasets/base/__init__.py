from .resize_dataset import ResizeDataset
from .sliding_dataset import SlidingWindowDataset
from .padding_dataset import PaddingDataset
from .util import filter_same_annotation, filter_same_annotation_decls
from .resize_multi_dataset import ResizeMultiDataset

__all__ = ["ResizeDataset", "SlidingWindowDataset", "PaddingDataset", "filter_same_annotation", "filter_same_annotation_decls", "ResizeMultiDataset"]

# There are three types of dataset preprocessing in TAD.
# 1. ResizeDataset: for each video, resize the video to a fixed length. It is commonly used in ActivityNet / HACS dataset.
# 2. SlidingWindowDataset: for each video, split the video into several windows. Such as used by G-TAD / BMN in THUMOS dataset.
# 3. PaddingDataset: for each video, pad the video to a fixed length. If the video is longer than the fixed length, then random cut a clip from the video. Such as used by ActionFormer / TriDet in THUMOS / EPIC-KITCHENS dataset.
