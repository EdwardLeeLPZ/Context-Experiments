import os
import json
import numpy as np
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

id_map = {0: 24, 1: 25, 2: 26, 3: 27, 4: 28, 5: 31, 6: 32, 7: 33, 8: 0}
reversed_id_map = {24: 0, 25: 1, 26: 2, 27: 3, 28: 4, 31: 5, 32: 6, 33: 7, 0: 8, 29: 8, 30: 8}


def generate_gt(input_dir, gt_dir, output_dir, dataset_type='train', label_format='binary'):
    n = 1
    for file in os.listdir(os.path.join(input_dir, dataset_type)):
        print(f'precessing image No.{n}')
        n += 1

        json_dir = os.path.join(input_dir, dataset_type, file)

        with open(json_dir) as json_file:
            data = json.load(json_file)

        image_name = data["image_id"]
        cityname = image_name.split('_')[0]
        gtname = '_'.join(image_name.split('_')[0:3])
        gt_instanceIds_image = Image.open(os.path.join(
            gt_dir, dataset_type, cityname, gtname + '_gtFine_instanceIds.png'))
        gt_instanceIds = np.asarray(gt_instanceIds_image)

        for i, instance in enumerate(data["instances"]):
            x1, y1, x2, y2 = instance["box"]
            gt_ids = np.unique(gt_instanceIds[y1:y2, x1:x2])
            gt_ids = (gt_ids // 1000).tolist()
            while gt_ids and gt_ids[0] == 0:
                gt_ids.pop(0)
            gt_ids = [reversed_id_map[id] for id in gt_ids]

            correctness = (instance["category"] in gt_ids)

            if label_format == 'binary':
                if correctness:
                    gt_ids.remove(instance["category"])
                have_context = (gt_ids != [])

                label_data = np.array([int(correctness), int(have_context)])
            elif label_format == 'multi-class':
                area = (x2 - x1) * (y2 - y1)
                gt_box = (gt_instanceIds[y1:y2, x1:x2] // 1000)
                cnt = Counter(gt_box.flatten())
                label_data = np.zeros(10)
                label_data[0] = correctness
                for id in cnt:
                    percentage = cnt[id] / area
                    label_data[reversed_id_map[id] + 1] = percentage
            else:
                err_msg = "Unrecognized label format: {}.".format(str(label_format))
                raise ValueError(err_msg)

            if not os.path.exists(os.path.join(output_dir, dataset_type)):
                os.makedirs(os.path.join(output_dir, dataset_type))

            np.save(file=os.path.join(output_dir, dataset_type, image_name.split(
                    '.')[0] + f"_{i}.npy"), arr=label_data)


def simplify_dataset(feature_input_dir, feature_output_dir):
    n = 1
    for file in os.listdir(feature_input_dir):
        print(f'precessing image No.{n}')
        n += 1

        feature_dir = os.path.join(feature_input_dir, file)

        with open(feature_dir) as feature_file:
            feature = json.load(feature_file)

        image_name = feature["image_id"]

        for i, instance in enumerate(feature['instances']):
            feature_data = np.array(instance['feature_map'])
            category = instance["category"]
            x1, y1, x2, y2 = instance["box"]
            proposal_idx = instance["proposal_idx"]

            if not os.path.exists(feature_output_dir):
                os.makedirs(feature_output_dir)

            instance_name = "_".join((image_name.split('.')[0], str(category), str(
                x1), str(y1), str(x2), str(y2), str(proposal_idx))) + ".npy"

            np.save(file=os.path.join(feature_output_dir, instance_name), arr=feature_data)


def merge_labelspace(label):
    label = label.type(torch.long)

    label = torch.where(label < 0, 114, label)  # other vehicles

    label = torch.where(label <= 6, 108, label)  # void
    label = torch.where(label <= 10, 109, label)  # flat
    label = torch.where(label <= 16, 110, label)  # construction
    label = torch.where(label <= 20, 111, label)  # object
    label = torch.where(label <= 22, 112, label)  # nature
    label = torch.where(label == 23, 113, label)  # sky

    label = torch.where(label == 24, 100, label)  # person
    label = torch.where(label == 25, 101, label)  # rider
    label = torch.where(label == 26, 102, label)  # car
    label = torch.where(label == 27, 103, label)  # truck
    label = torch.where(label == 28, 104, label)  # bus
    label = torch.where(label == 31, 105, label)  # train
    label = torch.where(label == 32, 106, label)  # motorcycle
    label = torch.where(label == 33, 107, label)  # bicycle

    label = torch.where(label < 100, 114, label)  # other vehicles

    label = label - 100
    return label


class PoolingFeatureDataset(Dataset):
    def __init__(self, feature_dir, label_dir, model_type, zoom_rate=0, simple_labelspace=False):
        assert os.path.exists(feature_dir)
        assert os.path.exists(label_dir)

        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.file_list = os.listdir(self.feature_dir)
        self.model_type = model_type
        self.zoom_rate = zoom_rate
        self.simple_labelspace = simple_labelspace

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.model_type == 'binary' or self.model_type == 'multi-class':
            file_name = self.file_list[idx]
            feature = np.load(file=os.path.join(self.feature_dir, file_name))
            feature = torch.tensor(feature).float()

            label = np.load(file=os.path.join(self.label_dir, file_name))
            label = torch.tensor(label).float()
            return feature, label
        elif self.model_type == 'seg':
            file_name = self.file_list[idx]
            feature = np.load(file=os.path.join(self.feature_dir, file_name))
            feature = torch.tensor(feature).float()

            city_name = file_name.split('_')[0]
            category, x1, y1, x2, y2 = [int(i) for i in file_name.split('.')[0].split('_')[4:9]]
            category = torch.tensor(id_map[category])

            gt_instanceIds = Image.open(os.path.join(
                self.label_dir, city_name, '_'.join(file_name.split('_')[0:3]) + '_gtFine_labelIds.png'))
            gt_instanceIds = np.asarray(gt_instanceIds)

            x1 = int(max(0, x1 - self.zoom_rate * (x2 - x1) * 0.5))
            x2 = int(min(gt_instanceIds.shape[1], x2 + self.zoom_rate * (x2 - x1) * 0.5))
            y1 = int(max(0, y1 - self.zoom_rate * (y2 - y1) * 0.5))
            y2 = int(min(gt_instanceIds.shape[0], y2 + self.zoom_rate * (y2 - y1) * 0.5))
            bbox = torch.tensor([x1, y1, x2, y2])

            label = torch.tensor(gt_instanceIds[y1:y2, x1:x2]).float()
            if self.simple_labelspace:
                label = merge_labelspace(label)
            return [feature, bbox, category], label
        elif self.model_type == 'rgb':
            file_name = self.file_list[idx]
            feature = np.load(file=os.path.join(self.feature_dir, file_name))
            feature = torch.tensor(feature).float()

            city_name = file_name.split('_')[0]
            category, x1, y1, x2, y2 = [int(i) for i in file_name.split('.')[0].split('_')[4:9]]
            category = torch.tensor(id_map[category])

            image = Image.open(os.path.join(
                self.label_dir, city_name, '_'.join(file_name.split('_')[0:3]) + '_leftImg8bit.png'))
            image = np.asarray(image)

            x1 = int(max(0, x1 - self.zoom_rate * (x2 - x1) * 0.5))
            x2 = int(min(image.shape[1], x2 + self.zoom_rate * (x2 - x1) * 0.5))
            y1 = int(max(0, y1 - self.zoom_rate * (y2 - y1) * 0.5))
            y2 = int(min(image.shape[0], y2 + self.zoom_rate * (y2 - y1) * 0.5))
            bbox = torch.tensor([x1, y1, x2, y2])

            label = torch.tensor(image[y1:y2, x1:x2].transpose((2, 0, 1))).float()
            return [feature, bbox, category], label
        elif self.model_type == 'gray':
            file_name = self.file_list[idx]
            feature = np.load(file=os.path.join(self.feature_dir, file_name))
            feature = torch.tensor(feature).float()

            city_name = file_name.split('_')[0]
            category, x1, y1, x2, y2 = [int(i) for i in file_name.split('.')[0].split('_')[4:9]]
            category = torch.tensor(id_map[category])

            image = Image.open(os.path.join(
                self.label_dir, city_name, '_'.join(file_name.split('_')[0:3]) + '_leftImg8bit.png'))
            image_transforms = transforms.Compose([transforms.Grayscale(1)])
            image = image_transforms(image)    
            image = np.asarray(image)

            x1 = int(max(0, x1 - self.zoom_rate * (x2 - x1) * 0.5))
            x2 = int(min(image.shape[1], x2 + self.zoom_rate * (x2 - x1) * 0.5))
            y1 = int(max(0, y1 - self.zoom_rate * (y2 - y1) * 0.5))
            y2 = int(min(image.shape[0], y2 + self.zoom_rate * (y2 - y1) * 0.5))
            bbox = torch.tensor([x1, y1, x2, y2])

            label = torch.tensor(image[y1:y2, x1:x2]).unsqueeze(0).float()
            return [feature, bbox, category], label
        elif self.model_type == 'segprd':
            file_name = self.file_list[idx]
            feature = np.load(file=os.path.join(self.feature_dir, file_name))
            feature = torch.tensor(feature).float()

            city_name = file_name.split('_')[0]
            category, x1, y1, x2, y2 = [int(i) for i in file_name.split('.')[0].split('_')[4:9]]
            category = torch.tensor(id_map[category])
            bbox = torch.tensor([x1, y1, x2, y2])

            gt_instanceIds = Image.open(os.path.join(
                self.label_dir, city_name, '_'.join(file_name.split('_')[0:3]) + '_gtFine_labelIds.png'))
            gt_instanceIds = np.asarray(gt_instanceIds)

            x1_z = int(max(0, x1 - self.zoom_rate * (x2 - x1) * 0.5))
            x2_z = int(min(gt_instanceIds.shape[1], x2 + self.zoom_rate * (x2 - x1) * 0.5))
            y1_z = int(max(0, y1 - self.zoom_rate * (y2 - y1) * 0.5))
            y2_z = int(min(gt_instanceIds.shape[0], y2 + self.zoom_rate * (y2 - y1) * 0.5))

            relative_coord = torch.tensor([x1-x1_z, y1-y1_z, x2-x1_z, y2-y1_z])

            zoom_label = torch.tensor(gt_instanceIds[y1_z:y2_z, x1_z:x2_z]).float()

            return [feature, bbox, category], [zoom_label, relative_coord]

# if __name__ == "__main__":
#     generate_gt(input_dir='/lhome/peizhli/datasets/cityscapes/pooling_feature',
#                 gt_dir='/lhome/peizhli/datasets/cityscapes/gtFine',
#                 output_dir='/lhome/peizhli/datasets/cityscapes/pooling_gt(multi-class classifier)',
#                 dataset_type='val',
#                 label_format='multi-class')

#     simplify_dataset(feature_input_dir='/lhome/peizhli/datasets/cityscapes/pooling_feature/train',
#                      feature_output_dir='/lhome/peizhli/datasets/cityscapes/pooling_feature_sim/train')
