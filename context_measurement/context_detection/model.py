import torch
import torch.nn as nn
import torch.nn.functional as F


class ContectClassifier(nn.Module):
    def __init__(self, input_size=[256, 7, 7], output_size=1):
        super(ContectClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size[0]*self.input_size[1]*self.input_size[2], 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output=self.classifier(input)
        return output


class MulticlassContectClassifier(nn.Module):
    def __init__(self, input_size=[256, 7, 7], output_size=34):
        super(MulticlassContectClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.multiclassifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size[0]*self.input_size[1]*self.input_size[2], 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size),
            nn.Softmax(),
        )

    def forward(self, input):
        output=self.multiclassifier(input)
        return output


class ContextDecoder(nn.Module):
    def __init__(self, input_size=256, output_size=34):
        super(ContextDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.upsampling_stack = nn.Sequential(
            nn.ConvTranspose2d(self.input_size, self.input_size, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.input_size, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_size, kernel_size=3, stride=2),
            nn.ReLU(),
        )  # H * W: 75 * 75

        self.convolution_stack = nn.Sequential(
            nn.Conv2d(self.output_size, self.output_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.output_size, self.output_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.output_size, self.output_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.output_size, self.output_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.output_size, self.output_size, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, input, bbox):
        output = self.upsampling_stack(input)
        x1, y1, x2, y2 = bbox[0, 0], bbox[0, 1], bbox[0, 2], bbox[0, 3]
        output = F.interpolate(output, size=[y2 - y1, x2 - x1], mode='bilinear')
        output = self.convolution_stack(output)
        return output


class ImageDecoder(nn.Module):
    def __init__(self, input_size=256, output_size=3):
        super(ImageDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.upsampling_stack = nn.Sequential(
            nn.ConvTranspose2d(self.input_size, self.input_size, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.input_size, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )  # H * W: 155 * 155

        self.convolution_stack = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, self.output_size, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input, bbox):
        output = self.upsampling_stack(input)
        x1, y1, x2, y2 = bbox[0, 0], bbox[0, 1], bbox[0, 2], bbox[0, 3]
        output = F.interpolate(output, size=[y2 - y1, x2 - x1], mode='bilinear')
        output = self.convolution_stack(output)
        output = output * 255
        return output


class ContextPyramidDecoder(nn.Module):
    def __init__(self, zoom_rate, input_size=256, output_size=34):
        super(ContextPyramidDecoder, self).__init__()
        self.zoom_rate = zoom_rate
        self.input_size = input_size
        self.output_size = output_size
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.w = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        self.preprocessing_stack = nn.Sequential(
            nn.ConvTranspose2d(self.input_size, self.input_size, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(self.input_size, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )  # H * W: 15 * 15

        self.upsampling_stack_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )  # H * W: 31 * 31

        self.upsampling_stack_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )  # H * W: 63 * 63

        self.upsampling_stack_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )  # H * W: 127 * 127

        self.upsampling_stack_4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )  # H * W: 255 * 255

        self.upsampling_stack_5 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )  # H * W: 511 * 511

        self.upsampling_stack_6 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )  # H * W: 1023 * 1023

        self.zooming_stack_25 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=25, padding=25),
            nn.ReLU(),
        )  # H * W: h * w

        self.zooming_stack_50 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=50, padding=50),
            nn.ReLU(),
        )  # H * W: h * w

        self.zooming_stack_100 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=100, padding=100),
            nn.ReLU(),
        )  # H * W: h * w

        self.zooming_stack_200 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=200, padding=200),
            nn.ReLU(),
        )  # H * W: h * w

        self.zooming_stack_400 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=400, padding=400),
            nn.ReLU(),
        )  # H * W: h * w

        self.zooming_stack_800 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=800, padding=800),
            nn.ReLU(),
        )  # H * W: h * w

        self.convolution_stack = nn.Sequential(
            nn.Conv2d(64, self.output_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.output_size, self.output_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.output_size, self.output_size, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, input, bbox):
        output = self.preprocessing_stack(input)

        x1, y1, x2, y2 = bbox[0, 0], bbox[0, 1], bbox[0, 2], bbox[0, 3]
        t0 = torch.tensor([y2 - y1, x2 - x1])
        k = torch.log2(torch.max(torch.tensor([x2 - x1, y2 - y1]))) - 3
        k = k.int()

        if k >= 1:
            output = self.upsampling_stack_1(output)
            if k >= 2:
                output = self.upsampling_stack_2(output)
                if k >= 3:
                    output = self.upsampling_stack_3(output)
                    if k >= 4:
                        output = self.upsampling_stack_4(output)
                        if k >= 5:
                            output = self.upsampling_stack_5(output)
                            if k >= 6:
                                output = self.upsampling_stack_6(output)

        output = F.interpolate(output, size=[y2 - y1, x2 - x1], mode='bilinear')
        regular_output = output

        x1_z = int(max(0, x1 - self.zoom_rate * (x2 - x1) * 0.5))
        x2_z = int(min(2048, x2 + self.zoom_rate * (x2 - x1) * 0.5))
        y1_z = int(max(0, y1 - self.zoom_rate * (y2 - y1) * 0.5))
        y2_z = int(min(1024, y2 + self.zoom_rate * (y2 - y1) * 0.5))
        t = torch.max(torch.tensor([x1 - x1_z, x2_z - x2, y1 - y1_z, y2_z - y2]))
        if t <= 100:
            m = 25
        elif 100 < t <= 200:
            m = 50
        elif 200 < t <= 300:
            m = 100
        elif 300 < t <= 600:
            m = 200
        elif 600 < t <= 800:
            m = 400
        else:
            m = 800

        output = nn.ReplicationPad2d((x1 - x1_z, x2_z - x2, y1 - y1_z, y2_z - y2))(output)
        while t > 0:
            if m == 25:
                output = self.zooming_stack_25(output)
            elif m == 50:
                output = self.zooming_stack_50(output)
            elif m == 100:
                output = self.zooming_stack_100(output)
            elif m == 200:
                output = self.zooming_stack_200(output)
            elif m == 400:
                output = self.zooming_stack_400(output)
            else:
                output = self.zooming_stack_800(output)
            t -= m
            # print(output.shape)

        output[:, :, y1 - y1_z:y2 - y1_z, x1 - x1_z:x2 - x1_z] *= self.w
        output[:, :, y1 - y1_z:y2 - y1_z, x1 - x1_z:x2 - x1_z] += (1.0-self.w) * regular_output 
        # print('--------------------------------------------')

        output = self.convolution_stack(output)

        return output
