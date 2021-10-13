import json
import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from posixpath import ismount
from PIL import Image
from cityscapesscripts.helpers.labels import Label, labels as cs_labels

context_judgement = 'Distance'  # 'IOA' or 'Distance'
# context_IOU_threshold = 0.1
context_IOA_threshold = 0.1
context_distance_threshold = 50

# threshold_list = [0,25,50,75,100,125,150,175,200]
threshold_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
image_number = 500
id_map = {0: 24000, 1: 25000, 2: 26000, 3: 27000, 4: 28000, 5: 31000, 6: 32000, 7: 33000, 8: 0}
reversed_id_map = {24: 0, 25: 1, 26: 2, 27: 3, 28: 4, 31: 5, 32: 6, 33: 7, 0: 8, 29: 8, 30: 8}
reversed_id_map_onehot = {24: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float),
                          25: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float),
                          26: np.array([0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float),
                          27: np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float),
                          28: np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float),
                          29: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float),
                          30: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float),
                          31: np.array([0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float),
                          32: np.array([0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float),
                          33: np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float),
                          0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float)}


def main(args):
    input_dir = args.input_dir
    gt_dir = args.gt_dir
    output_dir = args.output_dir
    analyse_postNMS_confidence_entropy = args.analyse_postNMS_confidence_entropy
    analyse_postNMS_APs_with_GT = args.analyse_postNMS_APs_with_GT
    analyse_confidence_entropy_with_GT = args.analyse_confidence_entropy_with_GT

    json_dir = os.path.join(input_dir, 'json')
    mask_dir = os.path.join(input_dir, 'mask')

    postNMS_confidence_entropy_records = []
    postNMS_APs_with_GT_records = []
    postNMS_APs_with_GT_FN_records = []
    confidence_entropy_with_GT = []

    for idx, json_file_name in enumerate(os.listdir(json_dir)):
        sys.stdout.flush()
        sys.stdout.write("Analyzing image No.%s.\r" % (idx + 1))

        full_json_file_path = os.path.join(json_dir, json_file_name)
        with open(full_json_file_path) as json_file:
            data = json.load(json_file)

        proposals = data['proposals']
        refined_boxes = data['refined_boxes']

        basename = '.'.join(json_file_name.split('.')[:-1])
        single_frame_mask_dir = os.path.join(mask_dir, basename)
        mask_list = os.listdir(single_frame_mask_dir)
        assert len(mask_list) == len(refined_boxes)

        for mask_name in mask_list:
            id = int(mask_name.split('.')[0])
            mask_image = Image.open(os.path.join(single_frame_mask_dir, mask_name))
            # convert image to numpy array
            mask = np.asarray(mask_image)
            refined_boxes[id]['mask'] = mask

        gtname = '_'.join(basename.split('_')[0:3])
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

        if analyse_postNMS_confidence_entropy:
            postNMS_confidence_entropy_records.extend(
                postNMS_confidence_entropy_analysis(refined_boxes, basename))

        if analyse_postNMS_APs_with_GT:
            temp=postNMS_APs_with_GT_analysis(refined_boxes, gts, basename)
            postNMS_APs_with_GT_records.extend(temp[0])
            postNMS_APs_with_GT_FN_records.extend(temp[1])

        if analyse_confidence_entropy_with_GT:
            confidence_entropy_with_GT.extend(
                confidence_entropy_with_GT_analysis(refined_boxes, gts, basename))

        if idx >= image_number - 1:
            break

    sys.stdout.flush()

    output(output_dir, postNMS_confidence_entropy_records, postNMS_APs_with_GT_records,
           postNMS_APs_with_GT_FN_records, confidence_entropy_with_GT)


def postNMS_confidence_entropy_analysis(refined_boxes, imagename):
    records = []
    for foreground in refined_boxes:
        record = {
            'image': imagename,
            'id': foreground['id'],
            'have_context': False,
            'category': foreground['softmax'].index(max(foreground['softmax'])),
            'confidence': max(foreground['softmax']),
            'entropy': -np.sum(np.array(foreground['softmax']) * np.log(np.array(foreground['softmax']))),
            'relevant_context': [],
            'irrelevant_context': [],
        }
        ymin, xmin = int(round(foreground['coords'][0])), int(round(foreground['coords'][1]))
        ymax, xmax = int(round(foreground['coords'][2])), int(round(foreground['coords'][3]))

        # calculate foreground coverage
        foreground_overlap = foreground['mask'][xmin:xmax, ymin:ymax].astype('bool')
        foreground_coverage = np.sum(foreground_overlap.astype('float')
                                     ) / ((xmax - xmin) * (ymax - ymin))
        record['foreground_coverage'] = foreground_coverage
        for context in refined_boxes:
            if foreground['id'] == context['id']:
                continue
            # calculate IOU
            context_ymin, context_xmin = int(
                round(context['coords'][0])), int(round(context['coords'][1]))
            context_ymax, context_xmax = int(
                round(context['coords'][2])), int(round(context['coords'][3]))
            C_ymin, C_xmin = min(ymin, context_ymin), min(xmin, context_xmin)
            C_ymax, C_xmax = max(ymax, context_ymax), max(xmax, context_xmax)
            intersection = foreground['mask'][C_xmin:C_xmax, C_ymin:C_ymax].astype(
                'bool') * context['mask'][C_xmin:C_xmax, C_ymin:C_ymax].astype('bool')
            union = foreground['mask'][C_xmin:C_xmax, C_ymin:C_ymax].astype(
                'bool') + context['mask'][C_xmin:C_xmax, C_ymin:C_ymax].astype('bool')
            area=foreground['mask'][C_xmin:C_xmax, C_ymin:C_ymax].astype('bool')
            # IOU = np.sum(intersection.astype('float')) / (np.sum(union.astype('float')) + 1e-8)
            IOA = np.sum(intersection.astype('float')) / (np.sum(area.astype('float')) + 1e-8)
            # if IOU > context_IOU_threshold:
            #     record['have_context'] = True
            #     if foreground['softmax'].index(max(foreground['softmax'])) == context['softmax'].index(max(context['softmax'])):
            #         record['relevant_context'].append({
            #             'IOU': IOU,
            #         })
            #     else:
            #         record['irrelevant_context'].append({
            #             'category': context['softmax'].index(max(context['softmax'])),
            #             'IOU': IOU,
            #         })
            if IOA > context_IOA_threshold:
                record['have_context'] = True
                if foreground['softmax'].index(max(foreground['softmax'])) == context['softmax'].index(max(context['softmax'])):
                    record['relevant_context'].append({
                        'IOA': IOA,
                    })
                else:
                    record['irrelevant_context'].append({
                        'category': context['softmax'].index(max(context['softmax'])),
                        'IOA': IOA,
                    })

        records.append(record)
    return records


def postNMS_APs_with_GT_analysis(refined_boxes, gts, imagename):
    records = []
    TP_list = []

    _, gt_instanceIds, _, _ = gts
    gt_instanceIds = gt_instanceIds * (gt_instanceIds > 33).astype('int')

    instances = {}
    for id in np.unique(gt_instanceIds):
        if id == 0:
            continue
        x, y = np.where(gt_instanceIds == id)
        instances[id] = [np.mean(x), np.mean(y)]

    for id in np.unique(gt_instanceIds):
        if id == 0:
            continue
        x, y = np.where(gt_instanceIds == id)
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        record = {
            'image': imagename,
            'id': id,
            'category': id // 1000,
            'have_context': False,
            'bbox': [xmin, ymin, xmax, ymax],
            'relevant_context': [],
            'irrelevant_context': [],
        }

        for context_id in np.unique(gt_instanceIds[xmin:xmax + 1, ymin:ymax + 1]):
            if context_id == 0 or context_id == id:
                continue
            area = (xmax - xmin + 1) * (ymax - ymin + 1)
            IOA = np.sum((gt_instanceIds[xmin:xmax + 1, ymin:ymax + 1]
                          == context_id).astype('float')) / area
            distance = np.sqrt(np.square(
                instances[id][0] - instances[context_id][0]) + np.square(instances[id][1] - instances[context_id][1]))
            if (context_judgement == 'IOA' and IOA > context_IOA_threshold) or (context_judgement == 'Distance' and distance < context_distance_threshold):
                record['have_context'] = True
                if context_id // 1000 == id // 1000:
                    record['relevant_context'].append({
                        'IOA': IOA,
                        'distance_rate': distance,
                    })
                else:
                    record['irrelevant_context'].append({
                        'category': context_id // 1000,
                        'IOA': IOA,
                        'distance_rate': distance,
                    })

        record['refined_boxes'] = {'TP': False, 'id': -1, 'IOU':
                                   - 1.0, 'confidence': 0.0, 'entropy': float('inf')}
        for foreground in refined_boxes:
            foreground_ymin = int(max(round(foreground['coords'][0]), 0))
            foreground_xmin = int(max(round(foreground['coords'][1]), 0))
            foreground_ymax = int(
                min(round(foreground['coords'][2]), foreground['mask'].shape[1] - 1))
            foreground_xmax = int(
                min(round(foreground['coords'][3]), foreground['mask'].shape[0] - 1))
            if foreground_ymin > ymax or foreground_xmin > xmax or foreground_ymax < ymin or foreground_xmax < xmin:
                continue
            intersection = (min(foreground_xmax, xmax) - max(foreground_xmin, xmin)) * \
                (min(foreground_ymax, ymax) - max(foreground_ymin, ymin))
            union = (ymax - ymin) * (xmax - xmin) + (foreground_ymax - foreground_ymin) * \
                (foreground_xmax - foreground_xmin) - intersection
            IOU = intersection / union
            if IOU > 0.5 and reversed_id_map[record['category']] == foreground['softmax'].index(max(foreground['softmax'])) and max(foreground['softmax']) > record['refined_boxes']['confidence']:
                record['refined_boxes']['TP'] = True
                if TP_list != [] and record['refined_boxes']['id'] != -1:
                    TP_list.remove(record['refined_boxes']['id'])
                record['refined_boxes']['IOU'] = IOU
                record['refined_boxes']['id'] = foreground['id']
                TP_list.append(record['refined_boxes']['id'])
                record['refined_boxes']['confidence'] = max(foreground['softmax'])
                record['refined_boxes']['entropy'] = - \
                    np.sum(np.array(foreground['softmax'])
                           * np.log(np.array(foreground['softmax'])))

        records.append(record)

    FN_records = []
    for rb in refined_boxes:
        if rb['id'] not in TP_list:
            FN_record = {'image': imagename, 'rb_category': rb['softmax'].index(
                max(rb['softmax'])), 'softmax': rb['softmax'], 'confidence': max(rb['softmax'])}
            FN_records.append(FN_record)

    return records, FN_records


def confidence_entropy_with_GT_analysis(refined_boxes, gts, imagename):
    records = []

    _, gt_instanceIds, _, _ = gts
    gt_instanceIds = gt_instanceIds * (gt_instanceIds > 33).astype('int')

    instances = {}
    for id in np.unique(gt_instanceIds):
        if id == 0:
            continue
        x, y = np.where(gt_instanceIds == id)
        instances[id] = [np.mean(x), np.mean(y)]

    for id in np.unique(gt_instanceIds):
        if id == 0:
            continue
        x, y = np.where(gt_instanceIds == id)
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        record = {
            'image': imagename,
            'id': id,
            'category': id // 1000,
            'have_context': False,
            'bbox': [xmin, ymin, xmax, ymax],
            'relevant_context': [],
            'irrelevant_context': [],
        }

        for context_id in np.unique(gt_instanceIds[xmin:xmax + 1, ymin:ymax + 1]):
            if context_id == 0 or context_id == id:
                continue
            area = (xmax - xmin + 1) * (ymax - ymin + 1)
            IOA = np.sum((gt_instanceIds[xmin:xmax + 1, ymin:ymax + 1]
                          == context_id).astype('float')) / area
            distance = np.sqrt(np.square(
                instances[id][0] - instances[context_id][0]) + np.square(instances[id][1] - instances[context_id][1]))
            if (context_judgement == 'IOA' and IOA > context_IOA_threshold) or (context_judgement == 'Distance' and distance < context_distance_threshold):
                record['have_context'] = True
                if context_id // 1000 == id // 1000:
                    record['relevant_context'].append({
                        'IOA': IOA,
                        'distance': distance,
                    })
                else:
                    record['irrelevant_context'].append({
                        'category': context_id // 1000,
                        'IOA': IOA,
                        'distance': distance,
                    })

        record['refined_boxes'] = {'TP': False, 'id': -1, 'IOU':
                                   - 1.0, 'confidence': 0.0, 'entropy': float('inf')}
        IOU_list = []
        confidence_list = []
        entropy_list = []
        for foreground in refined_boxes:
            foreground_ymin = int(max(round(foreground['coords'][0]), 0))
            foreground_xmin = int(max(round(foreground['coords'][1]), 0))
            foreground_ymax = int(
                min(round(foreground['coords'][2]), foreground['mask'].shape[1] - 1))
            foreground_xmax = int(
                min(round(foreground['coords'][3]), foreground['mask'].shape[0] - 1))
            if foreground_ymin > ymax or foreground_xmin > xmax or foreground_ymax < ymin or foreground_xmax < xmin:
                continue
            intersection = (min(foreground_xmax, xmax) - max(foreground_xmin, xmin)) * \
                (min(foreground_ymax, ymax) - max(foreground_ymin, ymin))
            union = (ymax - ymin) * (xmax - xmin) + (foreground_ymax - foreground_ymin) * \
                (foreground_xmax - foreground_xmin) - intersection
            IOU = intersection / union
            # take weighted average
            if IOU > 0.5 and reversed_id_map[record['category']] == foreground['softmax'].index(max(foreground['softmax'])):
                record['refined_boxes']['TP'] = True
                record['refined_boxes']['id'] = foreground['id']
                IOU_list.append(IOU)
                confidence_list.append(max(foreground['softmax']) * IOU)
                entropy_list.append(-np.sum(np.array(foreground['softmax'])
                                    * np.log(np.array(foreground['softmax']))) * IOU)
            # # only take the refined box with highest confidence
            # if IOU > 0.5 and reversed_id_map[record['category']] == foreground['softmax'].index(max(foreground['softmax'])) and max(foreground['softmax']) > record['refined_boxes']['confidence']:
            #     record['refined_boxes']['TP'] = True
            #     record['refined_boxes']['IOU'] = IOU
            #     record['refined_boxes']['id'] = foreground['id']
            #     record['refined_boxes']['confidence'] = max(foreground['softmax'])
            #     record['refined_boxes']['entropy'] = - \
            #         np.sum(np.array(foreground['softmax'])
            #                * np.log(np.array(foreground['softmax'])))
        if record['refined_boxes']['TP']:
            record['refined_boxes']['IOU'] = sum(IOU_list) / len(IOU_list)
            record['refined_boxes']['confidence'] = sum(
                confidence_list) / sum(IOU_list)
            record['refined_boxes']['entropy'] = sum(
                entropy_list) / sum(IOU_list)

        records.append(record)
    return records


def output(output_dir, postNMS_confidence_entropy_records=False, postNMS_APs_with_GT_records=False, postNMS_APs_with_GT_FN_records=False, confidence_entropy_with_GT=False):
    def calculate_weighted_average(value):
        i = 0
        while i < len(value):
            if value[i] != []:
                value[i] = sum(value[i]) / len(value[i])
                i += 1
            else:
                value.pop(i)
        value = sum(value) / (len(value) + 1e-8)
        return value

    def calculate_mAP(P_R_list):
        AP_list = [[], [], [], [], [], [], [], [], []]
        for i in range(len(P_R_list)):
            if P_R_list[i] == []:
                AP_list[i] = 0.0
                continue
            P_R_list[i].sort(key=lambda record: record[2], reverse=True)
            FN = P_R_list[i].count([0., 0., 0.])
            if FN != 0:
                P_R_list[i] = P_R_list[i][:-FN]
            if P_R_list[i] == []:
                AP_list[i] = 0.0
                continue
            P_R_list[i] = np.array(P_R_list[i])
            GT = np.sum(P_R_list[i][:, 0]) + FN
            for j in range(np.size(P_R_list[i], 0)):
                TP_j = np.sum(P_R_list[i][:j + 1, 0])
                FP_j = np.sum(P_R_list[i][:j + 1, 1])
                preciion = TP_j / (TP_j + FP_j + 1e-8)
                recall = TP_j / (GT + 1e-8)
                AP_list[i].append([preciion, recall])
            AP = 0.0
            for recall_threshold in range(0, 11):
                recall_threshold /= 10
                for j in range(len(AP_list[i])):
                    if AP_list[i][j][1] > recall_threshold:
                        AP += max(np.array(AP_list[i])[j:, 0])
                        break
            AP_list[i] = AP / 11 * 100
        mAP = sum(AP_list) / len(AP_list)
        return AP_list, mAP

    if postNMS_confidence_entropy_records:
        print("#-----POSTNMS CONFIDENCE UND ENTROPY ANALYSIS-----#")
        record_number = len(postNMS_confidence_entropy_records)
        print("The number of refined boxes is {}.".format(record_number))

        confi_w_context = [[], [], [], [], [], [], [], [], []]
        confi_wo_context = [[], [], [], [], [], [], [], [], []]
        entropy_w_context = [[], [], [], [], [], [], [], [], []]
        entropy_wo_context = [[], [], [], [], [], [], [], [], []]
        confi = []
        cover = []
        confi_w_relevant_context = []
        IOA_w_relevant_context = []
        confi_w_irrelevant_context = []
        IOA_w_irrelevant_context = []
        for record in postNMS_confidence_entropy_records:
            confi.append(record["confidence"])
            cover.append(record["foreground_coverage"])

            if record["have_context"]:
                confi_w_context[record["category"]].append(record["confidence"])
                entropy_w_context[record["category"]].append(record["entropy"])
                for cont in record["relevant_context"]:
                    confi_w_relevant_context.append(record["confidence"])
                    IOA_w_relevant_context.append(cont["IOA"])
                for cont in record["irrelevant_context"]:
                    confi_w_irrelevant_context.append(record["confidence"])
                    IOA_w_irrelevant_context.append(cont["IOA"])
            else:
                confi_wo_context[record["category"]].append(record["confidence"])
                entropy_wo_context[record["category"]].append(record["entropy"])

        n_w_context = sum([len(i) for i in confi_w_context])
        n_wo_context = sum([len(i) for i in confi_wo_context])
        print(f"The number of instances with context is {n_w_context}")
        print(f"The number of instances without context is {n_wo_context}")

        ave_confi_w_context = calculate_weighted_average(confi_w_context)
        ave_confi_wo_context = calculate_weighted_average(confi_wo_context)
        print(
            f"The average confidence of instances with context objects is {ave_confi_w_context*100:2.2f}%.")
        print(
            f"The average confidence of instances without context objects is {ave_confi_wo_context*100:2.2f}%.")

        ave_entropy_w_context = calculate_weighted_average(entropy_w_context)
        ave_entropy_wo_context = calculate_weighted_average(entropy_wo_context)
        print(
            f"The average entropy of instances with context objects is {ave_entropy_w_context:.4f}.")
        print(
            f"The average entropy of instances without context objects is {ave_entropy_wo_context:.4f}.")

        if not os.path.exists(os.path.join(output_dir, 'postNMS_confidence_entropy')):
            os.makedirs(os.path.join(output_dir, 'postNMS_confidence_entropy'))

        plt.figure(1)
        plt.scatter(cover, confi, color='red', s=0.5)
        plt.title('The Relationship between Confidence and Foreground Coverage',
                  fontsize=10, fontweight='medium')
        plt.xlabel('Foreground Coverage')
        plt.ylabel('Confidence')
        plt.savefig(os.path.join(output_dir, 'postNMS_confidence_entropy',
                    f'foreground_coverage_confidence_{context_IOA_threshold}.jpg'))
        plt.close(1)

        plt.figure(2)
        plt.scatter(IOA_w_relevant_context, confi_w_relevant_context,
                    color='red', marker='x', s=1, label='with Relevant Context')
        plt.scatter(IOA_w_irrelevant_context, confi_w_irrelevant_context,
                    color='blue', marker='+', s=1, label='with Irrelevant Context')
        plt.title('The Relationship between Confidence and IOU',
                  fontsize=10, fontweight='medium')
        plt.xlabel('IOU')
        plt.ylabel('Confidence')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_dir, 'postNMS_confidence_entropy',
                    f'IOU_confidence_{context_IOA_threshold}.jpg'))
        plt.close(2)

    if postNMS_APs_with_GT_records:
        print("#-----POSTNMS AP WITH GROUND TRUTH ANALYSIS-----#")
        record_number = len(postNMS_APs_with_GT_records)
        print("The number of instances is {}.".format(record_number))

        IOU_w_context = [[], [], [], [], [], [], [], [], []]
        IOU_wo_context = [[], [], [], [], [], [], [], [], []]
        entropy_w_context = [[], [], [], [], [], [], [], [], []]
        entropy_wo_context = [[], [], [], [], [], [], [], [], []]
        P_R_list_w_context = [[], [], [], [], [], [], [], [], []]
        P_R_list_wo_context = [[], [], [], [], [], [], [], [], []]
        for record in postNMS_APs_with_GT_records:
            if record['refined_boxes']['TP']:
                if record['have_context']:
                    IOU_w_context[reversed_id_map[record["category"]]].append(
                        record['refined_boxes']['IOU'])
                    entropy_w_context[reversed_id_map[record["category"]]].append(
                        record['refined_boxes']['entropy'])
                    P_R_list_w_context[reversed_id_map[record["category"]]].append(
                        [1, 0, record['refined_boxes']['confidence']])
                else:
                    IOU_wo_context[reversed_id_map[record["category"]]].append(
                        record['refined_boxes']['IOU'])
                    entropy_wo_context[reversed_id_map[record["category"]]].append(
                        record['refined_boxes']['entropy'])
                    P_R_list_wo_context[reversed_id_map[record["category"]]].append(
                        [1, 0, record['refined_boxes']['confidence']])
            else:
                if record['have_context']:
                    P_R_list_w_context[reversed_id_map[record["category"]]].append(
                        [0, 0, record['refined_boxes']['confidence']])
                else:
                    P_R_list_wo_context[reversed_id_map[record["category"]]].append(
                        [0, 0, record['refined_boxes']['confidence']])

        for FN_record in postNMS_APs_with_GT_FN_records:
            P_R_list_w_context[FN_record["rb_category"]].append(
                [0, 1, FN_record['confidence']])
            P_R_list_wo_context[FN_record["rb_category"]].append(
                [0, 1, FN_record['confidence']])

        ave_IOU_w_context = calculate_weighted_average(IOU_w_context)
        ave_IOU_wo_context = calculate_weighted_average(IOU_wo_context)
        print(
            f"The average IOU (between refined box and ground truth) with context objects is {ave_IOU_w_context*100:2.2f}%.")
        print(
            f"The average IOU (between refined box and ground truth) without context objects is {ave_IOU_wo_context*100:2.2f}%.")

        ave_entropy_w_context = calculate_weighted_average(entropy_w_context)
        ave_entropy_wo_context = calculate_weighted_average(entropy_wo_context)
        print(
            f"The average entropy of instances with context objects is {ave_entropy_w_context:.4f}.")
        print(
            f"The average entropy of instances without context objects is {ave_entropy_wo_context:.4f}.")

        AP_list_w_context, mAP_w_context = calculate_mAP(P_R_list_w_context)
        AP_list_wo_context, mAP_wo_context = calculate_mAP(P_R_list_wo_context)
        print(f"The mAP of instances with context objects is {mAP_w_context:2.2f}% with APs:")
        print("person\t|rider\t|car\t|truck\t|bus\t|train\t|motorcycle\t|bicycle\t|other\t")
        print("%2.2f\t|%2.2f\t|%2.2f\t|%2.2f\t|%2.2f\t|%2.2f\t|%2.2f\t\t|%2.2f\t\t|%2.2f\t" % (AP_list_w_context[0], AP_list_w_context[1], AP_list_w_context[
              2], AP_list_w_context[3], AP_list_w_context[4], AP_list_w_context[5], AP_list_w_context[6], AP_list_w_context[7], AP_list_w_context[8]))
        print(
            f"The mAP entropy of instances without context objects is {mAP_wo_context:2.2f}% with APs:")
        print("person\t|rider\t|car\t|truck\t|bus\t|train\t|motorcycle\t|bicycle\t|other\t")
        print("%2.2f\t|%2.2f\t|%2.2f\t|%2.2f\t|%2.2f\t|%2.2f\t|%2.2f\t\t|%2.2f\t\t|%2.2f\t" % (AP_list_wo_context[0], AP_list_wo_context[1], AP_list_wo_context[
              2], AP_list_wo_context[3], AP_list_wo_context[4], AP_list_wo_context[5], AP_list_wo_context[6], AP_list_wo_context[7], AP_list_wo_context[8]))

    if confidence_entropy_with_GT:
        print("#-----PRENMS CONFIDENCE UND ENTROPY WITH GROUND TRUTH ANALYSIS-----#")
        record_number = len(confidence_entropy_with_GT)
        print("The number of instances is {}.".format(record_number))

        confi_w_context = [[], [], [], [], [], [], [], [], []]
        confi_w_relevant_context = [[], [], [], [], [], [], [], [], []]
        confi_w_irrelevant_context = [[], [], [], [], [], [], [], [], []]
        confi_w_both_context = [[], [], [], [], [], [], [], [], []]
        confi_wo_context = [[], [], [], [], [], [], [], [], []]
        entropy_w_context = [[], [], [], [], [], [], [], [], []]
        entropy_w_relevant_context = [[], [], [], [], [], [], [], [], []]
        entropy_w_irrelevant_context = [[], [], [], [], [], [], [], [], []]
        entropy_w_both_context = [[], [], [], [], [], [], [], [], []]
        entropy_wo_context = [[], [], [], [], [], [], [], [], []]

        for record in confidence_entropy_with_GT:
            if record['refined_boxes']['TP']:
                if record['have_context']:
                    confi_w_context[reversed_id_map[record["category"]]].append(
                        record['refined_boxes']['confidence'])
                    entropy_w_context[reversed_id_map[record["category"]]].append(
                        record['refined_boxes']['entropy'])
                    if record['relevant_context'] != [] and record['irrelevant_context'] != []:
                        confi_w_both_context[reversed_id_map[record["category"]]].append(
                            record['refined_boxes']['confidence'])
                        entropy_w_both_context[reversed_id_map[record["category"]]].append(
                            record['refined_boxes']['entropy'])
                    elif record['relevant_context'] != [] and record['irrelevant_context'] == []:
                        confi_w_relevant_context[reversed_id_map[record["category"]]].append(
                            record['refined_boxes']['confidence'])
                        entropy_w_relevant_context[reversed_id_map[record["category"]]].append(
                            record['refined_boxes']['entropy'])
                    elif record['relevant_context'] == [] and record['irrelevant_context'] != []:
                        confi_w_irrelevant_context[reversed_id_map[record["category"]]].append(
                            record['refined_boxes']['confidence'])
                        entropy_w_irrelevant_context[reversed_id_map[record["category"]]].append(
                            record['refined_boxes']['entropy'])
                else:
                    confi_wo_context[reversed_id_map[record["category"]]].append(
                        record['refined_boxes']['confidence'])
                    entropy_wo_context[reversed_id_map[record["category"]]].append(
                        record['refined_boxes']['entropy'])

        n_w_context = sum([len(i) for i in confi_w_context])
        n_wo_context = sum([len(i) for i in confi_wo_context])
        print(f"The number of TP instances with context is {n_w_context}")
        print(f"The number of TP instances without context is {n_wo_context}")

        ave_confi_w_context = calculate_weighted_average(confi_w_context)
        ave_confi_w_relevant_context = calculate_weighted_average(confi_w_relevant_context)
        ave_confi_w_irrelevant_context = calculate_weighted_average(confi_w_irrelevant_context)
        ave_confi_w_both_context = calculate_weighted_average(confi_w_both_context)
        ave_confi_wo_context = calculate_weighted_average(confi_wo_context)
        print(
            f"The average confidence of instances with context objects is {ave_confi_w_context*100:2.2f}%.")
        print(
            f"The average confidence of instances with same context objects is {ave_confi_w_relevant_context*100:2.2f}%.")
        print(
            f"The average confidence of instances with other context objects is {ave_confi_w_irrelevant_context*100:2.2f}%.")
        print(
            f"The average confidence of instances with both context objects is {ave_confi_w_both_context*100:2.2f}%.")
        print(
            f"The average confidence of instances without context objects is {ave_confi_wo_context*100:2.2f}%.")

        ave_entropy_w_context = calculate_weighted_average(entropy_w_context)
        ave_entropy_w_relevant_context = calculate_weighted_average(entropy_w_relevant_context)
        ave_entropy_w_irrelevant_context = calculate_weighted_average(entropy_w_irrelevant_context)
        ave_entropy_w_both_context = calculate_weighted_average(entropy_w_both_context)
        ave_entropy_wo_context = calculate_weighted_average(entropy_wo_context)
        print(
            f"The average entropy of instances with context objects is {ave_entropy_w_context:.4f}.")
        print(
            f"The average entropy of instances with same context objects is {ave_entropy_w_relevant_context:.4f}.")
        print(
            f"The average entropy of instances with other context objects is {ave_entropy_w_irrelevant_context:.4f}.")
        print(
            f"The average entropy of instances with both context objects is {ave_entropy_w_both_context:.4f}.")
        print(
            f"The average entropy of instances without context objects is {ave_entropy_wo_context:.4f}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str,
                        default='/lhome/peizhli/projects/context_experiments/object_detection/outputs/eval_records')
    parser.add_argument("--gt_dir", type=str,
                        default='/lhome/peizhli/datasets/cityscapes/gtFine')
    parser.add_argument("--output_dir", type=str,
                        default='/lhome/peizhli/projects/context_experiments/object_detection/eval_results/postNMS_confidence_entropy_IOA')
    parser.add_argument("--analyse_postNMS_confidence_entropy", action='store_true', default=True)
    parser.add_argument("--analyse_postNMS_APs_with_GT", action='store_true', default=False)
    parser.add_argument("--analyse_confidence_entropy_with_GT", action='store_true', default=False)
    args = parser.parse_args()
    for context_IOA_threshold in threshold_list:
        print(f'Results with [context_IOA_threshold = {context_IOA_threshold}]')
        main(args)
