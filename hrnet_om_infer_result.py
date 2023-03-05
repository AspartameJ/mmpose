import os
import time
import json


def hrnet_om_infer_result(result_keypoints='om_work_dir/result_keypoints.json',
                        person_keypoints_val2017='data/coco/annotations/person_keypoints_val2017.json',
                        output_file='rank0.out.log'):
    log_dict = {}
    result_dict = {}
    with open(result_keypoints,'r',encoding='utf-8') as f:
        json_list = json.load(f)

        for result in json_list:
            points_len = len(result['keypoints'])
            format_kepoints = []

            for i in range(0, points_len, 3):
                format_kepoints.append(int('{:.0f}'.format(result['keypoints'][i])))
                format_kepoints.append(int('{:.0f}'.format(result['keypoints'][i+1])))

            if result['image_id'] in log_dict:
                log_dict[result['image_id']]['keypoints'].append(format_kepoints)
                log_dict[result['image_id']]['score'].append(result['score'])
            else:
                log_dict[result['image_id']] = {'id': result['image_id']}
                log_dict[result['image_id']]['keypoints'] = [format_kepoints]
                log_dict[result['image_id']]['score'] = [result['score']]
                log_dict[result['image_id']]['target'] = []

    with open(person_keypoints_val2017,'r',encoding='utf-8') as ff:
        json_list = json.load(ff)['annotations']

        for val in json_list:
            if (val['image_id'] in log_dict) and (val['num_keypoints'] > 0):
                points_len = len(val['keypoints'])
                format_kepoints = []

                for i in range(0, points_len, 3):
                    format_kepoints.append(val['keypoints'][i])
                    format_kepoints.append(val['keypoints'][i+1]) 

                log_dict[val['image_id']]['target'].append(format_kepoints)

    for v in log_dict.values():
        result_dict['event'] = 'INFER_RESULT'
        result_dict['value'] = v
        # print(result_dict)
    # json_str = json.dumps(log_dict)
    # with open(output_file, 'w') as fp:
    #     fp.write(json_str)
    log_dict.clear()
    return result_dict

if __name__ == '__main__':
    infer_result = hrnet_om_infer_result(result_keypoints='tmp_work_dir/result_keypoints.json')