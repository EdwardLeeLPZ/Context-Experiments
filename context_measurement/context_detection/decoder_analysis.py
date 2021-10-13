import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import correlate

import torch

softmax_threshold = 0.01
id_map = {0: 24, 1: 25, 2: 26, 3: 27, 4: 28, 5: 31, 6: 32, 7: 33, 8: 0}
reversed_id_map = {24: 0, 25: 1, 26: 2, 27: 3, 28: 4, 31: 5, 32: 6, 33: 7, 0: 8, 29: 8, 30: 8}
classname_map = {0: 'Person', 1: 'Rider', 2: 'Car', 3: 'Truck',
                 4: 'Bus', 5: 'Train', 6: 'Motorcycle', 7: 'Bicycle', 8: 'Other'}


def main(args):
    input_dir = args.input_dir
    feature_dir = args.feature_dir
    model_dir = args.model_dir
    zoom_rate = args.zoom_rate
    output_dir = args.output_dir

    json_dir = os.path.join(input_dir, 'json')
    object_context = select_context(json_dir)

    for object_id in range(8):
        context_id = object_context[object_id]

        model = torch.load(model_dir).to(device)
        model.eval()

        res = {'Object with Context': [], 'Object without Context': []}
        scatters = [[], []]

        for file_name in os.listdir(feature_dir):
            json_file_name = '_'.join(file_name.split('_')[0:4]) + '.json'
            category, x1, y1, x2, y2, proposal_idx = [
                int(i) for i in file_name.split('.')[0].split('_')[4:]]

            if category != object_id:
                continue

            feature = np.load(file=os.path.join(feature_dir, file_name))
            feature = torch.tensor(feature).float().unsqueeze(0).to(device)

            x1 = int(max(0, x1 - zoom_rate * (x2 - x1) * 0.5))
            x2 = int(min(2048, x2 + zoom_rate * (x2 - x1) * 0.5))
            y1 = int(max(0, y1 - zoom_rate * (y2 - y1) * 0.5))
            y2 = int(min(1024, y2 + zoom_rate * (y2 - y1) * 0.5))
            bbox = torch.tensor([x1, y1, x2, y2]).unsqueeze(0).to(device)

            pred = model(feature, bbox)
            pred = pred.argmax(1).squeeze().cpu().numpy()
            coverage = (pred == id_map[context_id]).astype(
                np.float).sum() / ((x2 - x1) * (y2 - y1) + 1e-16)

            full_json_file_path = os.path.join(json_dir, json_file_name)
            with open(full_json_file_path) as json_file:
                json_data = json.load(json_file)

            for rb in json_data['refined_boxes']:
                if rb['proposal_index'] == proposal_idx:
                    if rb['softmax'][context_id] >= softmax_threshold:
                        res['Object with Context'].append(coverage)
                    else:
                        res['Object without Context'].append(coverage)
                    if coverage > 0:
                        scatters[0].append(coverage)
                        scatters[1].append(rb['softmax'][context_id])
                    break

        res['Object with Context'] = sum(res['Object with Context']) / \
            (len(res['Object with Context']) + 1e-16)
        res['Object without Context'] = sum(
            res['Object without Context']) / (len(res['Object without Context']) + 1e-16)

        print('-------Average Coverage of Context in the Prediction (Object--{}; Conetxt--{})-------'.format(
            classname_map[object_id], classname_map[context_id]))
        print('Object with Context: {}%; Object without Context: {}%'.format(
            round(res['Object with Context'] * 100, 2), round(res['Object without Context'] * 100, 2)))

        covariance, correlation = calculate_correlation(scatters[0], scatters[1])
        print('Covariance Matrix:\n{}'.format(covariance))
        print('Correlation Coefficient:\n{}'.format(correlation))

        if not os.path.exists(os.path.join(output_dir)):
            os.makedirs(os.path.join(output_dir))

        plt.figure(1)
        plt.axes(xscale="log", yscale="log")
        plt.scatter(scatters[0], scatters[1], color='red', s=0.5)
        plt.title('The Relationship between Decoder Segmentation Coverage\n and Object Detection Softmax',
                  fontsize=10, fontweight='medium')
        plt.xlabel('Coverage of {}'.format(classname_map[context_id]))
        plt.ylabel('Softmax of {}'.format(classname_map[context_id]))
        plt.savefig(os.path.join(output_dir,
                    'Coverage_Softmax_{}_{}.jpg'.format(classname_map[object_id], classname_map[context_id])))
        plt.close(1)


def select_context(json_dir):
    record = [[0 for i in range(9)] for j in range(9)]
    for idx, json_file_name in enumerate(os.listdir(json_dir)):
        sys.stdout.flush()
        sys.stdout.write("Analyzing image No.%s.\r" % (idx + 1))

        full_json_file_path = os.path.join(json_dir, json_file_name)
        with open(full_json_file_path) as json_file:
            data = json.load(json_file)
        for rb in data['refined_boxes']:
            foreground_category = rb['softmax'].index(max(rb['softmax']))
            rb['softmax'][foreground_category] = 0
            for i, p in enumerate(rb['softmax']):
                if p >= softmax_threshold:
                    record[foreground_category][i] += 1
    sys.stdout.flush()

    ret = [0 for i in range(9)]
    for i in range(9):
        record[i][8] = 0
        ret[i] = record[i].index(max(record[i]))
    return ret


def calculate_correlation(x, y):
    x, y = np.array(x), np.array(y)
    covariance = np.cov(np.stack((x, y)))
    correlation = np.corrcoef(np.stack((x, y)))
    return covariance, correlation[0, 1]


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str,
                        default='/lhome/peizhli/projects/context_experiments/outputs/eval_records')
    parser.add_argument("--feature_dir", type=str,
                        default='/lhome/peizhli/datasets/cityscapes/pooling_feature_simple/val')
    parser.add_argument("--model_dir", type=str,
                        default='/lhome/peizhli/context-disentanglement/context_measurement/context_detection/output/experiment_4_decoder_z0.2/model_at_epoch_5.pth')
    parser.add_argument("--zoom_rate", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str,
                        default='/lhome/peizhli/projects/context_experiments/eval_results_decoder')
    args = parser.parse_args()
    main(args)
