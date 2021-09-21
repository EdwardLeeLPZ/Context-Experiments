import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable


class ContextCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, remove_foreground=False):
        super(ContextCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.remove_foreground = remove_foreground

    def forward(self, input, target, category):
        if self.remove_foreground:
            ignore_index = category.item()
            return F.cross_entropy(input, target, weight=self.weight, ignore_index=ignore_index)
        else:
            return F.cross_entropy(input, target, weight=self.weight)


class ContextFocalLoss(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(ContextFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target, category):
        N=input.shape[0]
        C=input.shape[1]
        input=F.softmax(input, dim=1)
        input=input.view(N, C, -1)
        P=input.shape[2]

        traget_onehot=Variable(input.new(N, C, P).fill_(0))
        ids=target.view(N, 1, -1)
        assert(ids.shape[2]==P), "The pixelnumber of the input and the target must be the same."
        traget_onehot.scatter_(dim=1, index=ids.data, value=1.)

        prob=(input*traget_onehot).sum(1).view(N, 1, -1)

        batch_loss=-torch.pow((1-prob), self.gamma)*prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class ContextPerceptualLoss(nn.Module):
    def __init__(self):
        super(ContextPerceptualLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vgg_model = nn.ModuleList(list(models.vgg16(pretrained=True).features)[:23]).to(device)
        for name, parameter in self.vgg_model.named_parameters():
            parameter.requries_grad = False

    def forward(self, input, target):
        input_image = F.interpolate(input, size=[224, 224], mode='bilinear')
        target_image = F.interpolate(target, size=[224, 224], mode='bilinear')
        if input_image.shape[1] == 1:
            input_image = torch.cat((input_image, input_image, input_image), 1)
        if target_image.shape[1] == 1:
            target_image = torch.cat((target_image, target_image, target_image), 1)

        input_perceptual_feature = self.extract_feature_style(input_image)
        target_perceptual_feature = self.extract_feature_style(target_image)
        feature_loss = F.mse_loss(input_perceptual_feature[8], target_perceptual_feature[8])
        return feature_loss

    def extract_feature_style(self, x):
        feature = {}
        for index, layer in enumerate(self.vgg_model):
            x = layer(x)
            if index in {3, 8, 15, 22}:  # relu1_2, relu2_2, relu3_3, relu4_3
                feature[index] = x
        return feature

    def calculate_style_loss(self, x, y):
        x=x.reshape((x.shape[0], x.shape[1], -1))
        y=y.reshape((y.shape[0], y.shape[1], -1))
        Gx = torch.bmm(x, x.transpose(1, 2))
        Gy = torch.bmm(y, y.transpose(1, 2))
        loss = F.mse_loss(Gx, Gy)
        return loss


class ContextFrequencyLoss(nn.Module):
    def __init__(self):
        super(ContextFrequencyLoss, self).__init__()

    def forward(self, input, target):
        input_image = F.interpolate(input, size=[224, 224], mode='bilinear')
        target_image = F.interpolate(target, size=[224, 224], mode='bilinear')
        if input_image.shape[1] == 1:
            input_image = torch.cat((input_image, input_image, input_image), 1)
        if target_image.shape[1] == 1:
            target_image = torch.cat((target_image, target_image, target_image), 1)

        input_frequency_feature = torch.fft.fft2(input_image)
        target_frequency_feature = torch.fft.fft2(target_image)
        input_frequency_feature[:, :, 0:50, 0:50] = 0
        target_frequency_feature[:, :, 0:50, 0:50] = 0
        frequency_loss = F.mse_loss(input_frequency_feature.real, target_frequency_feature.real) + \
            F.mse_loss(input_frequency_feature.imag, target_frequency_feature.imag)

        return frequency_loss