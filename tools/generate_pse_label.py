import os
import json
import argparse
import copy
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Test a Temporal Action Detector")
    parser.add_argument("--semi_anno_path", type=str, default="data/ablation_annos/new_i3d_0.1.json")
    parser.add_argument("--pse_anno_path", type=str, default="exps/omnitad/anet/i3d_label0.1_video0.1_point0.1/gpu1_id0/result_detection.json")
    parser.add_argument("--output_file", type=str, default="data/ablation_annos/i3d_pse_label0.1_video0.1_point0.1.json")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    semi_anno_path = args.semi_anno_path
    pse_anno_path = args.pse_anno_path

    with open(semi_anno_path, 'r') as f:
        semi_anno_dict = json.load(f)
    with open(pse_anno_path, 'r') as f:
        pse_anno_dict = json.load(f)
    
    vid_names = list(semi_anno_dict['database'].keys())
    pse_annos = pse_anno_dict['results']

    ignore_num = 0
    for vid_name in tqdm(vid_names):
        if semi_anno_dict['database'][vid_name]['subset'] in ['training_unlabel', 'training_video', 'training_point']:
            if vid_name in pse_annos.keys():
                tmp_item = copy.deepcopy(semi_anno_dict['database'][vid_name])

                pse_items = pse_annos[vid_name]
                new_annos = []
                for pse_item in pse_items:
                    if pse_item['score'] >= 0.3:
                        new_annos.append(pse_item)
                if len(new_annos) > 0:
                    # if len(new_annos) > 1:
                    #     new_annos = [new_annos[0]]
                    tmp_item['annotations'] = new_annos
                    tmp_item['subset'] = "training"

                    semi_anno_dict['database'][vid_name] = tmp_item
                else:
                    ignore_num += 1
    
    output_file = args.output_file
    with open(output_file, "w") as out:
        json.dump(semi_anno_dict, out, indent=4)

    print('done...', ignore_num)
