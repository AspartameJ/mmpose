# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from argparse import ArgumentParser

import mmcv

from mmpose.apis import init_pose_model
from mmpose.datasets import DatasetInfo
from mmpose.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import numpy as np
import torch
from mmpose.core.evaluation import (aggregate_scale, aggregate_stage_flip,
                                    flip_feature_maps, get_group_preds,
                                    split_ae_outputs)
from mmpose.core.post_processing import oks_nms
from mmpose.core.post_processing.group import HeatmapParser
from mmcv.runner import get_dist_info
from mmpose.datasets import build_dataset
import shutil

def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1

def main():
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--work-dir', help='the dir to save evaluation results')
    parser.add_argument(
        '--img-path',
        type=str,
        help='Path to an image file or a image folder.')
    parser.add_argument(
        '--resized-img-result',
        type=str,
        default='resized_imgs',
        help='Root of the resized img file. ')
    parser.add_argument(
        '--fliped-img-result',
        type=str,
        default='fliped_imgs',
        help='Root of the fliped img file. ')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--pose-nms-thr',
        type=float,
        default=0.9,
        help='OKS threshold for pose NMS')
    parser.add_argument(
        '--eval',
        default='mAP',
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    
    parser.add_argument('--out',default='eval_result.json', help='output result file')

    args = parser.parse_args()

    if os.path.exists(args.work_dir):
        shutil.rmtree(args.work_dir)
    os.makedirs(args.resized_img_root)

    # prepare image list
    if osp.isfile(args.img_path):
        image_list = [args.img_path]
    elif osp.isdir(args.img_path):
        image_list = [
            osp.join(args.img_path, fn) for fn in os.listdir(args.img_path)
            if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
        ]
    else:
        raise ValueError('Image path should be an image or image folder.'
                         f'Got invalid image path: {args.img_path}')

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())
    cfg = pose_model.cfg
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
        sigmas = getattr(dataset_info, 'sigmas', None)
        skeleton = getattr(dataset_info, 'skeleton', None)

    # optional
    return_heatmap = False
    results = []
    # process each image
    for image_name in mmcv.track_iter_progress(image_list):
        # # test a single image, with a list of bboxes.
        # pose_nms_thr=args.pose_nms_thr,
        # # get dataset info
        # pose_results = []
        # returned_outputs = []

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

        test_scale_factor = img_metas['test_scale_factor']

        onnx_result = {}
        scale_heatmaps_list = []
        scale_tags_list = []
        
        for idx, s in enumerate(sorted(test_scale_factor, reverse=True)):
            resized_result_filename = '{}_0.npy'.format(image_name.split('/')[-1].split('.')[0])
            resized_result_path = os.path.join(args.resized_img_result,resized_result_filename)

            output_result = np.load(resized_result_path)
            output_result = [torch.from_numpy(output_result)]  #numpy2tensor2list
            
            heatmaps, tags = split_ae_outputs(
                output_result, cfg.model.test_cfg['num_joints'],
                cfg.model.test_cfg['with_heatmaps'], cfg.model.test_cfg['with_ae'],
                cfg.model.test_cfg.get('select_output_index', range(len(output_result))))
                
            if cfg.model.test_cfg.get('flip_test', True):
                # use flip test
                fliped_result_filename = '{}_0.npy'.format(image_name.split('/')[-1].split('.')[0])
                fliped_result_path = os.path.join(args.fliped_img_result,fliped_result_filename)

                outputs_flipped = np.load(fliped_result_path)
                outputs_flipped = [torch.from_numpy(outputs_flipped)]  #numpy2tensor2list

                heatmaps_flipped, tags_flipped = split_ae_outputs(
                    outputs_flipped, cfg.model.test_cfg['num_joints'],
                    cfg.model.test_cfg['with_heatmaps'], cfg.model.test_cfg['with_ae'],
                    cfg.model.test_cfg.get('select_output_index',
                                        range(len(output_result))))

                heatmaps_flipped = flip_feature_maps(
                    heatmaps_flipped, flip_index=img_metas['flip_index'])
                if cfg.model.test_cfg['tag_per_joint']:
                    tags_flipped = flip_feature_maps(
                        tags_flipped, flip_index=img_metas['flip_index'])
                else:
                    tags_flipped = flip_feature_maps(
                        tags_flipped, flip_index=None, flip_output=True)

            else:
                heatmaps_flipped = None
                tags_flipped = None                
        ###############################################################
        aggregated_heatmaps = aggregate_stage_flip(
            heatmaps,
            heatmaps_flipped,
            index=-1,
            project2image=cfg.model.test_cfg['project2image'],
            size_projected=img_metas["base_size"],
            align_corners=cfg.model.test_cfg.get('align_corners', True),
            aggregate_stage='average',
            aggregate_flip='average')

        aggregated_tags = aggregate_stage_flip(
            tags,
            tags_flipped,
            index=-1,
            project2image=cfg.model.test_cfg['project2image'],
            size_projected=img_metas["base_size"],
            align_corners=cfg.model.test_cfg.get('align_corners', True),
            aggregate_stage='concat',
            aggregate_flip='concat')
        
        if s == 1 or len(test_scale_factor) == 1:
            if isinstance(aggregated_tags, list):
                scale_tags_list.extend(aggregated_tags)
            else:
                scale_tags_list.append(aggregated_tags)
            
        if isinstance(aggregated_heatmaps, list):
            scale_heatmaps_list.extend(aggregated_heatmaps)
        else:
            scale_heatmaps_list.append(aggregated_heatmaps)
            
        aggregated_heatmaps = aggregate_scale(
            scale_heatmaps_list,
            align_corners=cfg.model.test_cfg.get('align_corners', True),
            aggregate_scale='average')

        aggregated_tags = aggregate_scale(
            scale_tags_list,
            align_corners=cfg.model.test_cfg.get('align_corners', True),
            aggregate_scale='unsqueeze_concat')
            
        heatmap_size = aggregated_heatmaps.shape[2:4]
        tag_size = aggregated_tags.shape[2:4]
        if heatmap_size != tag_size:
            tmp = []
            for idx in range(aggregated_tags.shape[-1]):
                tmp.append(
                    torch.nn.functional.interpolate(
                        aggregated_tags[..., idx],
                        size=heatmap_size,
                        mode='bilinear',
                        align_corners=cfg.model.test_cfg.get('align_corners',
                                                        True)).unsqueeze(-1))
            aggregated_tags = torch.cat(tmp, dim=-1)

        # perform grouping
        parser = HeatmapParser(cfg.model.test_cfg)
        grouped, scores = parser.parse(aggregated_heatmaps,
                                        aggregated_tags,
                                        cfg.model.test_cfg['adjust'],
                                        cfg.model.test_cfg['refine'])

        preds = get_group_preds(
            grouped,
            img_metas['center'],
            img_metas['scale'], [aggregated_heatmaps.size(3),
                    aggregated_heatmaps.size(2)],
            use_udp=cfg.model.test_cfg.get('use_udp', False))
        image_paths = []
        image_paths.append(img_metas['image_file']) 

        if return_heatmap:
            output_heatmap = aggregated_heatmaps.detach().cpu().numpy()
        else:
            output_heatmap = None
        
        onnx_result['preds'] = preds
        onnx_result['scores'] = scores
        onnx_result['image_paths'] = image_paths
        onnx_result['output_heatmap'] = output_heatmap
        results.append(onnx_result)
        ################################################################
        # if return_heatmap:
        #     returned_outputs.append(onnx_result['output_heatmap'])
        # else:
        #     returned_outputs.append(None)
        
        # for idx, pred in enumerate(onnx_result['preds']):
        #     area = (np.max(pred[:, 0]) - np.min(pred[:, 0])) * (
        #         np.max(pred[:, 1]) - np.min(pred[:, 1]))
        #     pose_results.append({
        #         'keypoints': pred[:, :3],
        #         'score': onnx_result['scores'][idx],
        #         'area': area,
        #     })

        # # pose nms
        # score_per_joint = cfg.model.test_cfg.get('score_per_joint', False)
        # keep = oks_nms(
        #     pose_results,
        #     pose_nms_thr,
        #     sigmas,
        #     score_per_joint=score_per_joint)
        # pose_results = [pose_results[_keep] for _keep in keep]

    
    rank, _ = get_dist_info()
    eval_config = cfg.get('evaluation', {})
    eval_config = merge_configs(eval_config, dict(metric=args.eval))

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, os.path.join(args.work_dir, args.out))

        dataset_ = build_dataset(cfg.data.test, dict(test_mode=True))
        results = dataset_.evaluate(results, args.work_dir, **eval_config)
        for k, v in sorted(results.items()):
            print(f'{k}: {v}')    


if __name__ == '__main__':
    main()
