import os
import time
import json
from hrnet_om_infer_result import hrnet_om_infer_result

def hrnet_om_infer_end():
    ##########INFER_START############
    start_time_all = time.time()
    # 数据预处理
    os.system('python3 hrnet_preprocess.py \
        configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py \
        hrnet_w32_coco_512x512-bcb8c247_20200816.pth \
        --img-path data/coco/val2017')
    start_time_infer = time.time()
    # 模型推理
    os.system('python3 -m ais_bench \
        --model dynamic_hrnet.om \
        --input resized_imgs \
        --output output_dir \
        --output_dirname resized_img_postprocess \
        --outfmt NPY \
        --auto_set_dymdims_mode 1 & \
        python3 -m ais_bench \
        --model dynamic_hrnet.om \
        --input fliped_imgs \
        --output output_flip_dir \
        --output_dirname flip_img_postprocess \
        --outfmt NPY \
        --auto_set_dymdims_mode 1')
    end_time_infer = time.time()
    # 数据后处理和精度验证
    os.system('python3 hrnet_postprocess.py \
        configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py \
        hrnet_w32_coco_512x512-bcb8c247_20200816.pth \
        --img-path ./data/coco/val2017 \
        --resized-img-result ./output_dir/resized_img_postprocess \
        --fliped-img-result ./output_flip_dir/flip_img_postprocess \
        --work-dir ./om_work_dir \
        --eval mAP')
    end_time_all = time.time()
    ##########INFER_END############
    #从输出日志中获取AP和AR字段属性值
    with open('mAP.json', 'r', encoding='utf-8') as f:
        json_str = json.load(f)
        AP = json_str.get('AP')
        AR = json_str.get('AR')
    #读取预处理文件数
    files = os.listdir('./data/coco/val2017')
    #fps
    sample_num = len(files)
    all_time = end_time_all - start_time_all
    fps = sample_num / all_time
    #lantency
    latency = 1 / fps
    #打印log
    log_dict = {}
    end_dict = {}
    log_dict['start_time_all'] = '{:.0f}'.format(start_time_all)
    log_dict['start_time_infer'] = '{:.0f}'.format(start_time_infer)
    log_dict['end_time_infer'] = '{:.0f}'.format(end_time_infer)
    log_dict['end_time_all'] = '{:.0f}'.format(end_time_all)
    log_dict['sample_num'] = '{:.0f}'.format(sample_num)
    log_dict['infer_AP'] = '{:.4f}'.format(AP)
    log_dict['infer_AR'] = '{:.4f}'.format(AR)
    log_dict['samples/sec'] = '{:.4f}'.format(fps)
    log_dict['latency'] = '{:.4f}'.format(latency)
    end_dict['event'] = 'INFER_END'
    end_dict['value'] = log_dict

    # json_str = json.dumps(end_dict)
    # with open('rank0.out.log', 'w') as fp:
    #     fp.write(json_str)
    # log_dict.clear()
    return end_dict

def tmp_om_infer_end():
    ##########INFER_START############
    start_time_all = time.time()
    # 数据预处理
    os.system('python3 hrnet_preprocess.py \
        configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py \
        hrnet_w32_coco_512x512-bcb8c247_20200816.pth \
        --img-path data/tmp')
    start_time_infer = time.time()
    # 模型推理
    os.system('python3 -m ais_bench \
        --model dynamic_hrnet.om \
        --input resized_imgs \
        --output output_dir \
        --output_dirname resized_img_postprocess \
        --outfmt NPY \
        --auto_set_dymdims_mode 1 & \
        python3 -m ais_bench \
        --model dynamic_hrnet.om \
        --input fliped_imgs \
        --output output_flip_dir \
        --output_dirname flip_img_postprocess \
        --outfmt NPY \
        --auto_set_dymdims_mode 1')
    end_time_infer = time.time()
    # 数据后处理和精度验证
    os.system('python3 hrnet_postprocess.py \
        configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py \
        hrnet_w32_coco_512x512-bcb8c247_20200816.pth \
        --img-path ./data/tmp \
        --resized-img-result ./output_dir/resized_img_postprocess \
        --fliped-img-result ./output_flip_dir/flip_img_postprocess \
        --work-dir ./tmp_work_dir \
        --eval mAP')
    end_time_all = time.time()
    ##########INFER_END############
    #从输出日志中获取AP和AR字段属性值
    with open('mAP.json', 'r', encoding='utf-8') as f:
        json_str = json.load(f)
        AP = json_str.get('AP')
        AR = json_str.get('AR')
    #读取预处理文件数
    files = os.listdir('./data/tmp')\
    #fps
    sample_num = len(files)
    all_time = end_time_all - start_time_all
    fps = sample_num / all_time
    #lantency
    latency = 1 / fps
    #打印log
    log_dict = {}
    end_dict = {}
    result_dict = {}
    log_dict['start_time_all'] = '{:.0f}'.format(start_time_all)
    log_dict['start_time_infer'] = '{:.0f}'.format(start_time_infer)
    log_dict['end_time_infer'] = '{:.0f}'.format(end_time_infer)
    log_dict['end_time_all'] = '{:.0f}'.format(end_time_all)
    log_dict['sample_num'] = '{:.0f}'.format(sample_num)
    log_dict['infer_AP'] = '{:.4f}'.format(AP)
    log_dict['infer_AR'] = '{:.4f}'.format(AR)
    log_dict['samples/sec'] = '{:.4f}'.format(fps)
    log_dict['latency'] = '{:.4f}'.format(latency)
    end_dict['event'] = 'INFER_END'
    end_dict['value'] = log_dict

    # json_str = json.dumps(end_dict)
    # with open('rank0.out.log', 'w') as fp:
    #     fp.write(json_str)
    # log_dict.clear()
    return end_dict

if __name__ == '__main__':
    output_file = 'log/infer_{}.log'.format(0)
    if os.path.exists(output_file):
        output_file = 'log/infer_{}.log'.format(int(output_file[-5])+1)
        
    infer_end = hrnet_om_infer_end()
    infer_result = hrnet_om_infer_result(result_keypoints='om_work_dir/result_keypoints.json')
    
    infer_result.append(json.dumps(infer_end))
    with open(output_file,'w') as f:
        f.writelines(infer_result)
