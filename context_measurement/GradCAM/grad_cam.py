import os
import cv2
import json
import numpy as np
import torch
from torch import nn

classname_map = {0: 'Person', 1: 'Rider', 2: 'Car', 3: 'Truck',
                 4: 'Bus', 5: 'Train', 6: 'Motorcycle', 7: 'Bicycle', 8: 'Other'}


def softmax2d(x):
    y = np.exp(x)
    y = y / (np.sum(y) + 1e-8)
    return y


def normalization(x, method='minmax'):
    if method == 'minmax':
        x -= np.min(x)
        x /= (np.max(x) + 1e-32)
    elif method == 'gaussian':
        x -= np.mean(x)
        x /= (1 * np.std(x) + 1e-32)
        x *= 0.5
        x += 0.5
        x = np.clip(x, 0.0, 1.0)
    return x


def pooling_feature_output(cfg, pooling_feature, image_id, train=True):
    if train:
        json_file_path = os.path.join(cfg.OUTPUT_DIR, 'pooling_feature', 'train')
    else:
        json_file_path = os.path.join(cfg.OUTPUT_DIR, 'pooling_feature', 'val')
    if not os.path.exists(json_file_path):
        os.makedirs(json_file_path)

    json_file = pooling_feature

    with open(os.path.join(json_file_path, image_id.split('.')[0] + ".json"), 'w') as outfile:
        json.dump(json_file, outfile, indent=4)


class GradCAM(object):
    def __init__(self, cfg, net):
        self.cfg = cfg
        self.net = net
        self.layer_name = self._get_last_conv_name()
        self.feature = {}
        self.gradient = {}
        self.net.eval()  # sets the module in evaluation mode.
        self.handlers = []
        self._register_hook()

    def _get_last_conv_name(self):
        layer_name = {'backbone': None}
        for name, m in self.net.named_modules():
            if name == 'roi_heads.box_pooler':
                layer_name['pooler'] = name
        return layer_name

    def _get_features_hook(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.feature['pooler'] = output
            # print("pooler feature shape:{}".format(output.shape))

    def _get_grads_hook(self, module, input_grad, output_grad):
        if output_grad[0].shape[0] == 1000:
            self.gradient['pooler'] = output_grad[0]
            # print("pooler gradient shape:{}".format(output_grad[0].shape))

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name in list(self.layer_name.values()):
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def gen_cam(self, feature, gradient):
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # normalization
        cam = normalization(cam, 'gaussian')
        return cam

    def gen_cam_plus(self, feature, gradient):
        gradient = np.maximum(gradient, 0.)  # ReLU
        indicate = np.where(gradient > 0, 1., 0.)  # sign function
        norm_factor = np.sum(gradient, axis=(1, 2))  # [C] normalization
        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / \
                norm_factor[i] if norm_factor[i] > 0. else 0.  # avoiding devided by 0
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]
        weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha * ReLU(gradient)

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # normalization
        cam = normalization(cam, 'gaussian')
        return cam

    def gen_cam_pixel(self, feature, gradient):
        cam = feature * gradient  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # normalization
        cam = normalization(cam, 'gaussian')
        return cam

    def gen_cam_spacial(self, feature, gradient):
        weight = np.mean(gradient, axis=(0))  # [H,W]

        cam = feature * weight[np.newaxis, :, :]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # normalization
        cam = normalization(cam, 'gaussian')
        return cam

    def gen_mask_image(self, image, cam, box):
        x1, y1, x2, y2 = box
        instance = image[:, y1:y2, x1:x2]  # [C,H,W]
        instance = np.transpose(instance, (1, 2, 0))  # [H,W,C]

        mask = np.clip(cam * 255, 0, 255)
        mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)  # [H,W,C]

        mask_image = cv2.addWeighted(instance, 1, mask, 0.25, 0)
        return mask_image

    def __call__(self, inputs, index=0):
        with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=True) as prof:

            image = inputs[0]['image'].detach().cpu().numpy()
            file_name = inputs[0]['file_name']
            image_id = inputs[0]['image_id']

            print(f'Processing image: {image_id}')

            self.net.zero_grad()
            outputs = self.net.inference(inputs)  # output = results, proposals, features

            ret = {'image_id': image_id, 'file_name': file_name, 'instances': []}
            # p_feature_ret = {'image_id': image_id, 'instances': []}
            for index in range(len(outputs[0][0]['instances'].scores)):

                score = outputs[0][0]['instances'].scores[index]
                # box come from which proposal
                proposal_idx = outputs[0][0]['instances'].nms_survivors[index].detach(
                ).cpu().numpy()
                category = outputs[0][0]['instances'].pred_classes[index].detach().cpu().numpy()
                box = outputs[0][0]['instances'].pred_boxes.tensor[index].detach().cpu().numpy()
                box=box.astype(np.int32)
                x1, y1, x2, y2 = box

                # # record pooling features
                # p_feature_ret['instances'].append({
                #     'proposal_idx': proposal_idx.tolist(),
                #     'box': box.tolist(),
                #     'category': category.tolist(),
                #     'score': score.tolist(),
                #     'feature_map': self.feature['pooler'][proposal_idx].detach().cpu().numpy().astype(np.float16).tolist()
                # })

                if (x2 - x1) >= 50 and (y2 - y1) >= 50:
                    self.net.zero_grad()
                    score.backward(retain_graph=True)

                    # extract features and gradients of pooling layer
                    p_gradient = self.gradient['pooler'][proposal_idx].detach(
                    ).cpu().numpy()  # [1,C,H,W]
                    p_feature = self.feature['pooler'][proposal_idx].detach(
                    ).cpu().numpy()  # [1,C,H,W]

                    p_gradient = np.squeeze(p_gradient)  # [C,H,W]
                    p_feature = np.squeeze(p_feature)  # [C,H,W]

                    # generate backbone cam
                    cam = self.gen_cam_pixel(p_feature, p_gradient)
                    # cam = self.gen_cam_spacial(p_feature, p_gradient)
                    # cam = self.gen_cam(p_feature, p_gradient)
                    # cam = self.gen_cam_plus(p_feature, p_gradient)

                    # resize to original proposal shape
                    cam = cv2.resize(cam, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)

                    if cam.sum() == 0:
                        continue

                    # generate mask image
                    mask_image = self.gen_mask_image(image, cam, box)

                    ret['instances'].append({
                        # 'mask_image': mask_image,
                        # 'cam': cam,
                        'proposal_idx': proposal_idx,
                        'box': box,
                        'category': category,
                        'score': score
                    })

                    cv2.imwrite(os.path.join(self.cfg.OUTPUT_DIR, image_id.split(
                        '.')[0] + '_' + classname_map[int(category)] + '_' + str(index) + '.png'), mask_image)

                    torch.cuda.empty_cache()

            # # output pooling features
            # pooling_feature_output(self.cfg, p_feature_ret, image_id, train=True)

            torch.cuda.empty_cache()
        prof.export_chrome_trace(os.path.join(self.cfg.OUTPUT_DIR, 'gradcam_profile.json'))
        return ret
