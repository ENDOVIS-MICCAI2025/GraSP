#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import os
import logging
import numpy as np

from copy import deepcopy
from .surgical_dataset import SurgicalDataset
from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class Levis(SurgicalDataset):
    """
    PSI-AVA dataloader.
    """

    def __init__(self, cfg, split):
        self.dataset_name = "Levis"
        self.zero_fill = 5
        self.cfg = cfg
        self.heichole_videos = [f'video_{str(i).zfill(3)}' for i in range(102, 126)]
        super().__init__(cfg,split)
    
    def keyframe_mapping(self, video_idx, sec_idx, sec):
        #breakpoint()
        return sec_idx 
        
    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        # breakpoint()
        # Assuming self._image_paths is a list of lists of strings
        # for i, paths in enumerate(self._image_paths):
        #    self._image_paths[i] = [path.replace('\n', '') for path in paths]

        # Get the path of the middle frame 
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        video_name = self._video_idx_to_name[video_idx]
        img_type = 'png' if video_name in self.heichole_videos else 'jpg'

        complete_name = '{}/{}.{}'.format(video_name, str(sec).zfill(self.zero_fill), img_type)

        #TODO: REMOVE when all done
        folder_to_images = "/".join(self._image_paths[video_idx][0].split('/')[:-2])
        path_complete_name = os.path.join(folder_to_images,complete_name)

        found_idx = self._image_paths[video_idx].index(path_complete_name)

        assert path_complete_name == self._image_paths[video_idx][center_idx], f'Different paths {path_complete_name} & {self._image_paths[video_idx][center_idx]} & {sec_idx} & {sec}'
        assert found_idx == center_idx, f'Different indexes {found_idx} & {center_idx}'
        assert int(self._image_paths[video_idx][center_idx].split('/')[-1].split('_')[-1].replace('.'+img_type,''))==sec, f'Different {self._image_paths[video_idx][center_idx].split("/")[-1].replace("."+img_type,"")} {sec}'

        # Get the frame idxs for current clip.
        #breakpoint()

        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
            length=self._video_length, 
            online = self.cfg.DATA.ONLINE,
        )

        assert center_idx in seq, f'Center index {center_idx} not in sequence {seq}'
        clip_label_list = deepcopy(self._keyframe_boxes_and_labels[video_idx][sec_idx])
        assert len(clip_label_list) > 0

        # Get boxes and labels for current clip.
        boxes = []
        
        # Add labels depending on the task
        all_labels = {task:[] for task in self._region_tasks}

        for task in self._frame_tasks:
            assert all(label[task]==clip_label_list[0][task] for label in clip_label_list), f'Inconsistent {task} labels for frame {complete_name}: {[label[task] for label in clip_label_list]}'
            all_labels[task] = clip_label_list[0][task]

        extra_data = {}
        
        boxes = np.zeros((1, 4))
                
        # Load images of current clip.
        #breakpoint()
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.ENDOVIS_DATASET.IMG_PROC_BACKEND
        )
        
        # Preprocess images and boxes
        imgs, boxes = self._images_and_boxes_preprocessing_cv2(
            imgs, boxes=boxes
        )
        
        imgs = utils.pack_pathway_output(self.cfg, imgs)

        if self.cfg.NUM_GPUS>1:
            video_num = int(video_name.replace('video_',''))
            frame_identifier = [video_num,sec]
        else:
            frame_identifier = complete_name
        
        return imgs, all_labels, extra_data, frame_identifier
