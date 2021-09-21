import torch
import numpy as np


def DecoderMetrics(pred, label):
    pred = pred.argmax(1)
    accuracy = (pred == label).type(torch.float).mean((1, 2)).sum().item()

    pred_class = np.unique(pred.cpu().numpy())
    label_class = np.unique(label.cpu().numpy())
    precision = np.intersect1d(pred_class, label_class).shape[0] / pred_class.shape[0]
    recall = np.intersect1d(pred_class, label_class).shape[0] / label_class.shape[0]

    IOUs = []
    for id in np.intersect1d(pred_class, label_class):
        p = (pred == id).type(torch.float)
        l = (label == id).type(torch.float)
        i = (p * l).sum()
        u = ((p + l) > 0).type(torch.float).sum()
        iou = i / u
        IOUs.append(iou)
    ave_IOU = sum(IOUs) / (len(IOUs) + 1e-16)

    return accuracy, precision, recall, ave_IOU
