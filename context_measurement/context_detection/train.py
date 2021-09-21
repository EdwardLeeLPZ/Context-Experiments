import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from cityscapesscripts.helpers.labels import Label, labels as cs_labels

from dataset import PoolingFeatureDataset, merge_labelspace
from model import ContectClassifier, MulticlassContectClassifier, ContextDecoder, ImageDecoder, ContextPyramidDecoder
from loss import ContextCrossEntropyLoss, ContextFocalLoss, ContextPerceptualLoss, ContextFrequencyLoss
from metrics import DecoderMetrics


def train_step(dataloader, model, output_dir, model_type, loss_fn, optimizer, correct_only=True):
    size = len(dataloader.dataset)
    average_loss = 0
    for batch, (feature, label) in enumerate(dataloader):
        # Predict
        if model_type == 'binary' or model_type == 'multi-class':
            feature, label = feature.to(device), label.to(device)
            pred = model(feature)
            if correct_only:
                correct_instance_index = torch.squeeze(torch.nonzero(label[:, 0]))
                loss = loss_fn(pred[correct_instance_index], label[correct_instance_index, 1:])
            else:
                loss = loss_fn(pred, label[:, 1:])
        elif model_type == 'seg' or model_type == 'rgb' or model_type == 'gray':
            feature, bbox, category = feature
            feature, bbox, category, label = feature.to(device), bbox.to(
                device), category.to(device), label.to(device)
            if bbox[0, 2] - bbox[0, 0] <= 0 or bbox[0, 3] - bbox[0, 1] <= 0:
                continue
            pred = model(feature, bbox)
            if model_type == 'seg':
                loss = loss_fn(pred, label.long(), category)
            else:
                loss = loss_fn(pred, label)
        elif model_type == 'segprd':
            feature, bbox, category = feature
            zoom_label, relative_coord = label
            feature, bbox, category = feature.to(device), bbox.to(
                device), category.to(device)
            zoom_label, relative_coord = zoom_label.to(device), relative_coord.to(device)
            if bbox[0, 2] - bbox[0, 0] <= 0 or bbox[0, 3] - bbox[0, 1] <= 0:
                continue
            pred = model(feature, bbox)
            loss = loss_fn(pred, zoom_label.long(), category) + loss_fn(pred[:, :, relative_coord[0, 1]:relative_coord[0, 3], relative_coord[0, 0]:relative_coord[0, 2]], zoom_label[:, relative_coord[0, 1]:relative_coord[0, 3], relative_coord[0, 0]:relative_coord[0, 2]].long(), category)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        average_loss = 0.99 * average_loss + 0.01 * loss.item()

        if batch != 0 and batch % 1000 == 0:
            loss, current = average_loss, batch * len(feature)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            torch.save(model, output_dir + f'/temporary_model.pth')

    return loss


def test_step(dataloader, model, model_type, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, accuracy, precision, recall, ave_IOU = 0, 0, 0, 0, 0
    with torch.no_grad():
        for (feature, label) in dataloader:
            # Predict
            if model_type == 'binary':
                pred = model(feature)
                feature, label = feature.to(device), label.to(device)
                test_loss += loss_fn(pred, label[:, 1:]).item()
                pred = (pred >= 0.5).type(torch.float)
                accuracy += (pred == label[:, 1:]).type(torch.float).sum().item()
            elif model_type == 'multi-class':
                feature, label = feature.to(device), label.to(device)
                pred = model(feature)
                test_loss += loss_fn(pred, label[:, 1:]).item()
                pred = (pred >= 0.05).type(torch.float)
                label[:, 1:] = (label[:, 1:] >= 0.05).type(torch.float)
                accuracy += (pred == label[:, 1:]).type(torch.float).sum().item() / pred.shape[0]
            elif model_type == 'seg' or model_type == 'rgb' or model_type == 'gray':
                feature, bbox, category = feature
                feature, bbox, category, label = feature.to(device), bbox.to(
                    device), category.to(device), label.to(device)
                if bbox[0, 2] - bbox[0, 0] <= 0 or bbox[0, 3] - bbox[0, 1] <= 0:
                    continue
                pred = model(feature, bbox)
                if model_type == 'seg':
                    test_loss += loss_fn(pred, label.long(), category).item()
                    acc, pre, rec, iou = DecoderMetrics(pred, label)
                    accuracy, precision, recall, ave_IOU = accuracy + acc, precision + pre, recall + rec, ave_IOU + iou
                else:
                    test_loss += loss_fn(pred, label).item()
            elif model_type == 'segprd':
                feature, bbox, category = feature
                zoom_label, relative_coord = label
                feature, bbox, category = feature.to(device), bbox.to(
                    device), category.to(device)
                zoom_label, relative_coord = zoom_label.to(device), relative_coord.to(device)
                if bbox[0, 2] - bbox[0, 0] <= 0 or bbox[0, 3] - bbox[0, 1] <= 0:
                    continue
                pred = model(feature, bbox)
                test_loss += loss_fn(pred, zoom_label.long(), category) + loss_fn(pred[:, :, relative_coord[0, 1]:relative_coord[0, 3], relative_coord[0, 0]:relative_coord[0, 2]], zoom_label[:, relative_coord[0, 1]:relative_coord[0, 3], relative_coord[0, 0]:relative_coord[0, 2]].long(), category)
                acc, pre, rec, iou = DecoderMetrics(pred, zoom_label)
                accuracy, precision, recall, ave_IOU = accuracy + acc, precision + pre, recall + rec, ave_IOU + iou

    test_loss /= num_batches
    accuracy /= size
    precision /= size
    recall /= size
    ave_IOU /= size

    if model_type == 'seg' or model_type == 'segprd':
        print(
            f"Test Result: \n Accuracy: {(100*accuracy):>0.1f}%, Precision: {(100*precision):>0.1f}%, Recall: {(100*recall):>0.1f}%, Average IOU: {(100*ave_IOU):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    elif model_type == 'rgb' or model_type == 'gray':
        print(f"Test Result: \n Avg loss: {test_loss:>8f} \n")
    else:
        print(f"Test Result: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, accuracy, precision, recall, ave_IOU


def train(batch_size, epochs, feature_dir, label_dir, output_dir, model_type='binary', learning_rate=1e-4, correct_only=True, zoom_rate=0, remove_foreground=False, simple_labelspace=False, resume_dir=''):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    writer = SummaryWriter(output_dir)

    train_set = PoolingFeatureDataset(feature_dir=os.path.join(feature_dir, 'train'),
                                      label_dir=os.path.join(label_dir, 'train'), model_type=model_type, zoom_rate=zoom_rate, simple_labelspace=simple_labelspace)
    test_set = PoolingFeatureDataset(feature_dir=os.path.join(feature_dir, 'val'),
                                     label_dir=os.path.join(label_dir, 'val'), model_type=model_type, zoom_rate=zoom_rate, simple_labelspace=simple_labelspace)

    if resume_dir != '':
        print(f'Resume from model: {resume_dir}')
        model = torch.load(resume_dir).to(device)
    else:
        model = None

    if model_type == 'binary':
        if not model:
            model = ContectClassifier(input_size=[256, 7, 7]).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.BCELoss()
    elif model_type == 'multi-class':
        if not model:
            model = MulticlassContectClassifier(input_size=[256, 7, 7]).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.KLDivLoss()
    elif model_type == 'seg':
        if not model:
            if simple_labelspace:
                model = ContextDecoder(input_size=256, output_size=15).to(device)
            else:
                model = ContextDecoder(input_size=256, output_size=34).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = ContextFocalLoss(gamma=0.5)
        # loss_fn = ContextCrossEntropyLoss(remove_foreground=remove_foreground)
    elif model_type == 'rgb':
        if not model:
            model = ImageDecoder(input_size=256, output_size=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = ContextPerceptualLoss()
    elif model_type == 'gray':
        if not model:
            model = ImageDecoder(input_size=256, output_size=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = ContextPerceptualLoss()
    elif model_type == 'segprd':
        if not model:
            if simple_labelspace:
                model = ContextPyramidDecoder(zoom_rate=zoom_rate, input_size=256, output_size=15).to(device)
            else:
                model = ContextPyramidDecoder(zoom_rate=zoom_rate, input_size=256, output_size=34).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # loss_fn = ContextFocalLoss(gamma=0.5)
        loss_fn = ContextCrossEntropyLoss(remove_foreground=remove_foreground)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=8)

    print('Model Architecture:')
    print(model)
    print("Model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train_step(train_loader, model, output_dir, model_type,
                          loss_fn, optimizer, correct_only)

        test_loss, accuracy, precision, recall, ave_IOU = test_step(
            test_loader, model, model_type, loss_fn)

        writer.add_scalar('Image Train Loss', loss, t + 1)
        writer.add_scalar('Image Test Loss', test_loss, t + 1)
        if model_type == 'rgb':
            writer.add_scalar('Accuracy', accuracy * 100, t + 1)
        if model_type == 'seg' or model_type == 'segprd':
            writer.add_scalar('Accuracy', accuracy * 100, t + 1)
            writer.add_scalar('Precision', precision * 100, t + 1)
            writer.add_scalar('Recall', recall * 100, t + 1)
            writer.add_scalar('Average IOU of True Positive', ave_IOU * 100, t + 1)

        if (t + 1) % 1 == 0:
            torch.save(model, output_dir + f'/model_at_epoch_{t+1}.pth')
            print("Saved PyTorch Model to " + output_dir + f'/model_at_epoch_{t+1}.pth')

    print("Done!")


def evaluate(batch_size, feature_dir, label_dir, model_dir, model_type='binary', correct_only=True, zoom_rate=0, remove_foreground=False, simple_labelspace=False):
    if model_type == 'binary':
        test_set = PoolingFeatureDataset(feature_dir=os.path.join(feature_dir, 'val'),
                                         label_dir=os.path.join(label_dir, 'val'), model_type=model_type)
        print(f'Resume from model: {model_dir}')
        model = torch.load(model_dir).to(device)
        loss_fn = nn.BCELoss()
    elif model_type == 'multi-class':
        test_set = PoolingFeatureDataset(feature_dir=os.path.join(feature_dir, 'val'),
                                         label_dir=os.path.join(label_dir, 'val'), model_type=model_type)
        print(f'Resume from model: {model_dir}')
        model = torch.load(model_dir).to(device)
        loss_fn = nn.KLDivLoss()
    elif model_type == 'seg':
        test_set = PoolingFeatureDataset(feature_dir=os.path.join(feature_dir, 'val'),
                                         label_dir=os.path.join(label_dir, 'val'), model_type=model_type, zoom_rate=zoom_rate, simple_labelspace=simple_labelspace)
        print(f'Resume from model: {model_dir}')
        model = torch.load(model_dir).to(device)
        loss_fn = ContextCrossEntropyLoss(remove_foreground=remove_foreground)

    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=8)

    _, _, _, _, _ = test_step(test_loader, model, model_type, loss_fn)


def visualize(image_dir, gt_dir, data_dir, model_dir, save_dir, model_type='seg', zoom_rate=0, simple_labelspace=False):
    if not os.path.exists(os.path.join(save_dir, 'visualization')):
        os.makedirs(os.path.join(save_dir, 'visualization'))

    model = torch.load(model_dir).to(device)
    model.eval()
    print('Model Architecture:')
    print(model)

    id2color = {label.id: label.color for label in cs_labels}
    id2color_simple = {
        0: id2color[24], 1: id2color[25], 2: id2color[26], 3: id2color[27], 4: id2color[28], 5: id2color[31], 6: id2color[32],
        7: id2color[33], 8: id2color[0], 9: id2color[7], 10: id2color[11], 11: id2color[17], 12: id2color[21], 13: id2color[23], 14: id2color[29]
    }

    def to_RGB(image, colormap):
        RGB_image = np.zeros((image.shape[0], image.shape[1], 3))
        for i in colormap:
            RGB_image[np.where(image == i)] = colormap[i]
        return RGB_image

    i = 1
    for file in os.listdir(data_dir):
        if i > 20:
            break

        feature = np.load(file=os.path.join(data_dir, file))
        feature = torch.tensor(feature).unsqueeze(0).float().to(device)

        image_name = '_'.join(file.split('_')[0:3])
        city_name = image_name.split('_')[0]

        image = Image.open(os.path.join(image_dir, city_name, image_name + '_leftImg8bit.png'))
        if model_type == 'gray':
            image_transforms = transforms.Compose([transforms.Grayscale(1)])
            image = image_transforms(image)
        image = np.asarray(image)
        gt = Image.open(os.path.join(gt_dir, city_name, image_name + '_gtFine_labelIds.png'))
        gt = np.asarray(gt)

        x1, y1, x2, y2 = [int(i) for i in file.split('.')[0].split('_')[5:9]]
        bbox = torch.tensor([x1, y1, x2, y2]).unsqueeze(0).to(device)

        if x2 - x1 < 50 or y2 - y1 < 50:
            continue
    
        x1_z = int(max(0, x1 - zoom_rate * (x2 - x1) * 0.5))
        x2_z = int(min(image.shape[1], x2 + zoom_rate * (x2 - x1) * 0.5))
        y1_z = int(max(0, y1 - zoom_rate * (y2 - y1) * 0.5))
        y2_z = int(min(image.shape[0], y2 + zoom_rate * (y2 - y1) * 0.5))
        bbox_z = torch.tensor([x1, y1, x2, y2]).unsqueeze(0).to(device)

        print(f'Saving image No.{i}')

        if model_type == 'segprd':
            pred = model(feature, bbox)
        else:
            pred = model(feature, bbox_z)

        image = image[y1_z:y2_z, x1_z:x2_z]
        gt = gt[y1_z:y2_z, x1_z:x2_z]

        if model_type == 'seg' or model_type == 'segprd':
            pred = pred.argmax(1).squeeze().cpu().detach().numpy()
            if simple_labelspace:
                pred = to_RGB(pred, id2color_simple)
                gt = merge_labelspace(torch.tensor(gt)).numpy()
                gt = to_RGB(gt, id2color_simple)
            else:
                pred = to_RGB(pred, id2color)
                gt = to_RGB(gt, id2color)
        elif model_type == 'rgb':
            pred = pred.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
        elif model_type == 'gray':
            pred = pred.squeeze().cpu().detach().numpy()

        Image.fromarray(np.uint8(pred)).save(os.path.join(
            save_dir, 'visualization', image_name + f'_pred_{i}.png'))
        Image.fromarray(np.uint8(image)).save(os.path.join(
            save_dir, 'visualization', image_name + f'_image_{i}.png'))
        if model_type == 'seg' or model_type == 'segprd':
            Image.fromarray(np.uint8(gt)).save(os.path.join(
                save_dir, 'visualization', image_name + f'_gt_{i}.png'))

        i += 1


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # train(
    #     batch_size=1,
    #     epochs=10,
    #     feature_dir='/lhome/peizhli/datasets/cityscapes/pooling_feature_simple',
    #     label_dir='/lhome/peizhli/datasets/cityscapes/gtFine',
    #     # label_dir='/lhome/peizhli/datasets/cityscapes/leftImg8bit',
    #     output_dir='/lhome/peizhli/context-disentanglement/context_measurement/context_detection/output/experiment_9_decoder_pyramid_refined',
    #     model_type='segprd',
    #     learning_rate=1e-4,
    #     correct_only=False,
    #     zoom_rate=1.0,
    #     remove_foreground=False,
    #     simple_labelspace=False,
    #     # resume_dir='/lhome/peizhli/context-disentanglement/context_measurement/context_detection/output/experiment_6_decoder/initial_model.pth',
    # )

    # evaluate(
    #     batch_size=1,
    #     feature_dir='/lhome/peizhli/datasets/cityscapes/pooling_feature_simple',
    #     label_dir='/lhome/peizhli/datasets/cityscapes/gtFine',
    #     model_dir='/lhome/peizhli/context-disentanglement/context_measurement/context_detection/output/experiment_4_decoder_z0.2/model_at_epoch_5.pth',
    #     model_type='seg',
    #     correct_only=False,
    #     zoom_rate=0.2,
    #     remove_foreground=False,
    #     simple_labelspace=False,
    # )

    visualize(
        image_dir='/lhome/peizhli/datasets/cityscapes/leftImg8bit/val',
        gt_dir='/lhome/peizhli/datasets/cityscapes/gtFine/val',
        data_dir='/lhome/peizhli/datasets/cityscapes/pooling_feature_simple/val',
        model_dir='/lhome/peizhli/context-disentanglement/context_measurement/context_detection/output/experiment_9_decoder_pyramid_refined/model_at_epoch_4.pth',
        save_dir='/lhome/peizhli/context-disentanglement/context_measurement/context_detection/output/experiment_9_decoder_pyramid_refined',
        model_type='segprd',
        zoom_rate=1.0,
        simple_labelspace=False,
    )
