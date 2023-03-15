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
from mmpose.core.evaluation import (aggregate_scale, aggregate_stage_flip,
                                    flip_feature_maps, get_group_preds,
                                    split_ae_outputs)
from mmpose.core.post_processing.group import HeatmapParser
from ais_bench.infer.interface import InferSession


def infer_dymdims(ndata, model_path):
    device_id = 0
    session = InferSession(device_id, model_path)

    #ndata = np.zeros([1,3,224,224], dtype=np.float32)

    mode = "dymdims"
    outputs = session.infer([ndata], mode)
    #print("outputs:{} type:{}".format(outputs, type(outputs)))

    #print("dymdims infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))
    return outputs


def hrnet_preprocess(img_metas):
    aug_data = img_metas['aug_data']
    image_resized = aug_data[0].to('cpu')
    image_fliped = torch.flip(image_resized, [3]).numpy().dtype(np.float32)
    image_resized = image_resized.numpy().dtype(np.float32)

    return image_resized, image_fliped

def hrnet_postprocess(img_metas, cfg, image_resized_output, image_fliped_output):
    return_heatmap = False
    results = []

    test_scale_factor = img_metas['test_scale_factor']

    onnx_result = {}
    scale_heatmaps_list = []
    scale_tags_list = []
    
    for idx, s in enumerate(sorted(test_scale_factor, reverse=True)):
        output_result = [torch.from_numpy(image_resized_output)]  #numpy2tensor2list
        
        heatmaps, tags = split_ae_outputs(
            output_result, cfg.model.test_cfg['num_joints'],
            cfg.model.test_cfg['with_heatmaps'], cfg.model.test_cfg['with_ae'],
            cfg.model.test_cfg.get('select_output_index', range(len(output_result))))
            
        if cfg.model.test_cfg.get('flip_test', True):
            # use flip test
            outputs_flipped = [torch.from_numpy(image_fliped_output)]  #numpy2tensor2list

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
    return results

def main():
    """Visualize the demo images."""
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('pose_om', help='om file')
    parser.add_argument(
        '--img-path',
        type=str,
        help='Path to an image file or a image folder.')
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

    cfg = pose_model.cfg
    for image_name in mmcv.track_iter_progress(image_list):
    # for image_name in image_list:
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
        
        image_resized, image_fliped = hrnet_preprocess(img_metas)
        image_resized_output = infer_dymdims(image_resized, args.pose_om)
        image_fliped_output = infer_dymdims(image_fliped, args.pose_om)

        results = hrnet_postprocess(img_metas, cfg, image_resized_output, image_fliped_output)

        print(results)

if __name__ == '__main__':
    main()
