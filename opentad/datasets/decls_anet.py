import numpy as np
from .base import ResizeDataset, PaddingDataset, SlidingWindowDataset, filter_same_annotation_decls
from .builder import DATASETS


@DATASETS.register_module()
class DeclsAnetResizeDataset(ResizeDataset):
    def get_gt(self, video_info, thresh=0.01):
        gt_segment = []
        decls_label = []
        gt_label = []
        for anno in video_info["annotations"]:
            gt_start = float(anno["segment"][0])
            gt_end = float(anno["segment"][1])
            gt_scale = (gt_end - gt_start) / float(video_info["duration"])

            if (not self.filter_gt) or (gt_scale > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))
                decls_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
                decls_labels=np.array(decls_label, dtype=np.int32),
            )
            return filter_same_annotation_decls(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]

        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                resize_length=self.resize_length,
                sample_stride=self.sample_stride,
                # resize post process setting
                fps=-1,
                duration=float(video_info["duration"]),
                **video_anno,
            )
        )
        return results
