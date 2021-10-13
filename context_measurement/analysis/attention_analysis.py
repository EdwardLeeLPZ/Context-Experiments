import json
import argparse
import os
import sys
import cv2
import numpy as np
from PIL import Image
from torch._C import dtype

image_number = 500
id_map = {0: 24000, 1: 25000, 2: 26000, 3: 27000, 4: 28000, 5: 31000, 6: 32000, 7: 33000, 8: 0}
reversed_id_map = {24: 0, 25: 1, 26: 2, 27: 3, 28: 4, 31: 5, 32: 6, 33: 7, 0: 8, 29: 8, 30: 8}
category_name = {24: 'person', 25: 'rider', 26: 'car', 27: 'truck', 28: 'bus',
                 31: 'train', 32: 'motorcycle', 33: 'bicycle', 0: 'other', 29: 'other', 30: 'other'}
image_dir = '/lhome/peizhli/datasets/cityscapes/leftImg8bit/val'


def main(args):
    input_dir = args.input_dir
    gt_dir = args.gt_dir
    output_dir = args.output_dir
    analyse_feature_attention = args.analyse_feature_attention

    json_dir = os.path.join(input_dir, 'json')
    mask_dir = os.path.join(input_dir, 'mask')

    feature_attention_records = []

    for idx, json_file_name in enumerate(os.listdir(json_dir)):
        sys.stdout.flush()
        sys.stdout.write("Analyzing image No.%s.\r" % (idx + 1))

        full_json_file_path = os.path.join(json_dir, json_file_name)
        with open(full_json_file_path) as json_file:
            data = json.load(json_file)

        proposals = data['proposals']
        refined_boxes = data['refined_boxes']
        features = data['features']

        basename = json_file_name.split('.')[0]
        single_frame_mask_dir = os.path.join(mask_dir, basename)
        mask_list = os.listdir(single_frame_mask_dir)
        assert len(mask_list) == len(refined_boxes)

        for mask_name in mask_list:
            id = int(mask_name.split('.')[0])
            mask_image = Image.open(os.path.join(single_frame_mask_dir, mask_name))
            # convert image to numpy array
            mask = np.asarray(mask_image)
            refined_boxes[id]['mask'] = mask

        gtname = '_'.join(basename.split('_')[:-1])
        cityname = basename.split('_')[0]

        if os.path.exists(os.path.join(gt_dir, 'val', cityname)):
            gt_dir = os.path.join(gt_dir, 'val')
        elif os.path.exists(os.path.join(gt_dir, 'test', cityname)):
            gt_dir = os.path.join(gt_dir, 'test')
        else:
            err_msg = "Unrecognized city name: {}".format(cityname)
            raise ValueError(err_msg)

        gt_color_image = Image.open(os.path.join(gt_dir, cityname, gtname + '_gtFine_color.png'))
        gt_color = np.asarray(gt_color_image)
        gt_instanceIds_image = Image.open(os.path.join(
            gt_dir, cityname, gtname + '_gtFine_instanceIds.png'))
        gt_instanceIds = np.asarray(gt_instanceIds_image)
        gt_labelIds_image = Image.open(os.path.join(
            gt_dir, cityname, gtname + '_gtFine_labelIds.png'))
        gt_labelIds = np.asarray(gt_labelIds_image)
        with open(os.path.join(gt_dir, cityname, gtname + '_gtFine_polygons.json')) as json_file:
            gt_polygons = json.load(json_file)

        gts = (gt_color, gt_instanceIds, gt_labelIds, gt_polygons)
        gt_dir = args.gt_dir

        for feature in features:
            feature['feature_map'] = np.squeeze(np.array(feature['feature_map']))
            feature['feature_map'] = denosing(feature['feature_map'])

            stride = 2**int(feature['feature_map_name'][1])
            dsize = (feature['feature_map'].shape[1] * stride,
                     feature['feature_map'].shape[0] * stride)
            feature['feature_map'] = cv2.resize(
                feature['feature_map'], dsize, interpolation=cv2.INTER_LINEAR)

        if analyse_feature_attention:
            feature_attention_records.extend(feature_attention_analysis(features, gts, basename))

        if idx >= image_number - 1:
            break

    sys.stdout.flush()

    output(output_dir, feature_attention_records)


def denosing(map):
    filter_map = np.zeros(map.shape) + np.mean(map)
    filter_map = (map >= filter_map).astype(np.float)
    map = map * filter_map
    return map


def spatial_scale_assign(width, length):
    k0 = 4
    k = np.floor(k0 + np.log2(np.sqrt(width * length) / 244 + 1e-8))
    return np.clip(k, 2, 5).astype(np.int)


def softmax2d(x):
    y = np.exp(x)
    y = y / np.sum(y)
    return y


def kl_divergence(p, q):
    d = np.sum(p * (np.log2(p) - np.log2(q)))
    return d


def feature_attention_analysis(features, gts, imagename):
    records = []
    gt_color, gt_instanceIds, gt_labelIds, gt_polygons = gts

    for id in np.unique(gt_instanceIds):
        if id <= 33:
            continue
        x, y = np.where(gt_instanceIds == id)
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        record = {
            'image': imagename,
            'id': id,
            'category': id // 1000,
            'bbox': [xmin, ymin, xmax, ymax],
        }

        spatial_scale = spatial_scale_assign(ymax - ymin, xmax - xmin)
        attention_map = features[spatial_scale - 2]['feature_map'][xmin:xmax, ymin:ymax]
        record['attention_map'] = attention_map

        gt_map = (gt_instanceIds[xmin:xmax, ymin:ymax] == id).astype(np.float)
        kld = kl_divergence(softmax2d(gt_map), softmax2d(attention_map))
        record['kl_divergence'] = kld

        records.append(record)
    return records


def output(output_dir, feature_attention_records=False):
    if feature_attention_records:
        print("#-----FEATURE ATTENTION ANALYSIS-----#")
        record_number = len(feature_attention_records)
        print("The number of boxes is {}.".format(record_number))

        kld = [[], [], [], [], [], [], [], [], []]
        for record in feature_attention_records:
            kld[reversed_id_map[record["category"]]].append(record['kl_divergence'])

            if not os.path.exists(os.path.join(output_dir, 'attention_map', category_name[record["category"]])):
                os.makedirs(os.path.join(output_dir, 'attention_map',
                            category_name[record["category"]]))

            if (record['bbox'][2]-record['bbox'][0])>=256 or (record['bbox'][3]-record['bbox'][1])>=256:
                cityname = record['image'].split('_')[0]
                imagename = record['image'] + '.png'
                image = np.asarray(cv2.imread(os.path.join(image_dir, cityname, imagename)))
                image = image[record['bbox'][0]:record['bbox']
                            [2], record['bbox'][1]:record['bbox'][3], :]
                # attention_map = (record['attention_map'] - np.min(record['attention_map'])) / \
                #     (np.max(record['attention_map']) - np.min(record['attention_map']) + 1e-8) * 255
                attention_map = ((record['attention_map'] - np.mean(record['attention_map'])) / \
                    (np.std(record['attention_map']) + 1e-8) + 1) * 255 / 2
                attention_map = np.clip(attention_map, 0, 255)
                attention_map = cv2.applyColorMap(attention_map.astype(np.uint8), cv2.COLORMAP_JET)
                image = cv2.addWeighted(image, 1, attention_map, 0.2, 0)
                cv2.imwrite(os.path.join(output_dir, 'attention_map', category_name[record["category"]], str(
                    len(kld[reversed_id_map[record["category"]]])) + '.png'), image)

        for i in range(len(kld)):
            if kld[i] == []:
                kld[i] = 0.0
            else:
                kld[i] = sum(kld[i]) / len(kld[i])

        print(
            f"The mean KL-divergence of instances are:")
        print("person\t|rider\t|car\t|truck\t|bus\t|train\t|motorcycle\t|bicycle\t|other\t")
        print("%2.4f\t|%2.4f\t|%2.4f\t|%2.4f\t|%2.4f\t|%2.4f\t|%2.4f\t\t|%2.4f\t\t|%2.4f\t" %
              (kld[0], kld[1], kld[2], kld[3], kld[4], kld[5], kld[6], kld[7], kld[8]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str,
                        default='/lhome/peizhli/projects/context_experiments/outputs_feature/eval_records')
    parser.add_argument("--gt_dir", type=str,
                        default='/lhome/peizhli/datasets/cityscapes/gtFine')
    parser.add_argument("--output_dir", type=str,
                        default='/lhome/peizhli/projects/context_experiments/eval_results_feature')
    parser.add_argument("--analyse_feature_attention", action='store_true', default=True)
    args = parser.parse_args()
    main(args)
