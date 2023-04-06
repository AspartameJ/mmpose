# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import mmcv

from mmpose.apis import init_pose_model
from mmpose.datasets import DatasetInfo
from mmpose.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

import numpy as np
import torch
import shutil

def main():
    """Visualize the demo images."""
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--img-path',
        type=str,
        help='Path to an image file or a image folder.')
    parser.add_argument(
        '--resized-img-root',
        type=str,
        default='resized_imgs',
        help='Root of the resized img file. ')
    parser.add_argument(
        '--fliped-img-root',
        type=str,
        default='fliped_imgs',
        help='Root of the fliped img file. ')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')

    args = parser.parse_args()

    # prepare image list
    if os.path.isfile(args.img_path):
        image_list = [args.img_path]
    elif os.path.isdir(args.img_path):
        image_list = [
            os.path.join(args.img_path, fn) for fn in os.listdir(args.img_path)
            if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
        ]
    else:
        raise ValueError('Image path should be an image or image folder.'
                         f'Got invalid image path: {args.img_path}')

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        assert (dataset == 'BottomUpCocoDataset')
    else:
        dataset_info = DatasetInfo(dataset_info)
        dataset_name = dataset_info.dataset_name
        flip_index = dataset_info.flip_index
        skeleton = getattr(dataset_info, 'skeleton', None)

    if os.path.exists(args.resized_img_root):
        shutil.rmtree(args.resized_img_root)
    if os.path.exists(args.fliped_img_root):
        shutil.rmtree(args.fliped_img_root)
    os.makedirs(args.resized_img_root)
    os.makedirs(args.fliped_img_root)

    # resized_shape_map = {}
    # fliped_shape_map = {}
    # process each image

    def preprocess(image_list):
        for image_name in mmcv.track_iter_progress(image_list):
        # for image_name in image_list:
            cfg = pose_model.cfg
            device = next(pose_model.parameters()).device
            if device.type == 'cpu':
                device = -1

            # build the data pipeline
            test_pipeline = Compose(cfg.test_pipeline)

            # prepare data
            data = {
                'dataset': dataset_name,
                'ann_info': {
                    'image_size': np.array(cfg.data_cfg['image_size']),
                    'heatmap_size': cfg.data_cfg.get('heatmap_size', None),
                    'num_joints': cfg.data_cfg['num_joints'],
                    'flip_index': flip_index,
                    'skeleton': skeleton,
                }
            }
        
            if isinstance(image_name, np.ndarray):
                data['img'] = image_name
            else:
                data['image_file'] = image_name

            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            data = scatter(data, [device])[0]        

            ###############################################################
            img_metas = data["img_metas"] #forward_val(..,img_metas,..)
            img = data['img'] #forward_val(..,img,..)
            ###############################################################
            assert img.size(0) == 1
            assert len(img_metas) == 1

            img_metas = img_metas[0]
            aug_data = img_metas['aug_data']
            
            image_resized = aug_data[0].to(img.device)
            image_fliped = torch.flip(image_resized, [3]).numpy()
            image_resized = image_resized.numpy()

            # resized_shape_map[image_resized.shape] = 1
            # fliped_shape_map[image_fliped.shape] = 1

        # for key, value in resized_shape_map.items():
        #     print(key)
        # print('=====================================================')
        # for key, value in fliped_shape_map.items():
        #     print(key)

            npy_filename = '{}.npy'.format(image_name.split('/')[-1].split('.')[0])
            output_path_resized = os.path.join(args.resized_img_root, npy_filename)
            output_path_fliped = os.path.join(args.fliped_img_root, npy_filename)
            
            np.save(output_path_resized, image_resized)
            np.save(output_path_fliped, image_fliped)

    preprocess(image_list)



if __name__ == '__main__':
    main()
