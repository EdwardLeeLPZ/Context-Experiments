import torch
from torch import nn, Tensor
from torch.jit.annotations import BroadcastingList2
from torchvision.extension import _assert_has_ops
from torchvision.ops._utils import convert_boxes_to_roi_format, check_roi_boxes_shape


def embedding(z: Tensor, c: int = 16):
    assert isinstance(c, int), "The dimension c must be an integer."
    epsilon = torch.cat(
        (torch.sin(z / torch.tensor([1000**(i / c) for i in range(1, c, 2)])),
         torch.cos(z / torch.tensor([1000**(j / c) for j in range(0, c, 2)]))
         )
    )
    return epsilon


def box_embedding(boxes: Tensor, dimension: int = 512):
    _assert_has_ops()
    check_roi_boxes_shape(boxes)
    if not isinstance(boxes, torch.Tensor):
        rois = convert_boxes_to_roi_format(boxes)

    embedding_boxes = []
    for box in boxes:
        embedding_boxes.append([
            box[0].type(torch.int64),  # batch index
            torch.cat((
                embedding(box[1], c=dimension),  # x1
                embedding(box[2], c=dimension),  # y1
                embedding(box[3], c=dimension),  # x2
                embedding(box[4], c=dimension),  # y2
            ))
        ])

    return embedding_boxes


def positions_embedding(input_size: list, dimension: int = 512):
    embedding_positions = []

    for index in range(input_size[0]):
        pos_per_img = []
        for x in range(input_size[2]):
            pos_per_row = []
            for y in range(input_size[3]):
                pos_per_row.append(torch.cat((
                    embedding(x, c=dimension),  # x1
                    embedding(y, c=dimension),  # y1
                )).reshape(-1, 1).unsqueeze(0)
                )
            pos_per_row = torch.cat(pos_per_row, dim=0)
            pos_per_img.append(pos_per_row.unsqueeze(0))
        pos_per_img = torch.cat(pos_per_img, dim=0)
        embedding_positions.append(pos_per_img)

    return embedding_positions

# def weighted_sum(feature:Tensor, weights:Tensor):


class BoxTerm(nn.Module):
    def __init__(self, input_dimension: int = 512, output_dimension: int = 256, output_size: int = 7):
        super(BoxTerm, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.output_size = output_size
        self.V_box = nn.Parameter(torch.randn(self.input_dimension, 4 * self.input_dimension))
        self.W_box = nn.ParameterList([nn.Parameter(torch.randn(
            self.output_dimension, self.input_dimension)) for i in range(self.output_size**2)])

    def forward(self, rois: Tensor):
        output = []
        for i, [index, epsilon_box] in enumerate(box_embedding(rois, self.input_dimension)):
            box_output = [index]
            for W_box_k in self.W_box:
                box_output.append(torch.mm(W_box_k, torch.mm(
                    self.V_box, epsilon_box.reshape(-1, 1))))
            output.append(box_output)
        return output


class PositionTerm(nn.Module):
    def __init__(self, input_dimension: int = 512, output_dimension: int = 256):
        super(PositionTerm, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.W_im = nn.Parameter(torch.randn(self.output_dimension, 2 * self.input_dimension))

    def forward(self, feature_map_size: list):
        output = [torch.matmul(self.W_im, img) for img in positions_embedding(feature_map_size)]
        return output


class AppearanceTerm(nn.Module):
    def __init__(self, input_channel: int, output_size: int = 7):
        super(AppearanceTerm, self).__init__()
        self.input_channel = input_channel
        self.output_size = output_size
        self.conv = nn.Conv2d(self.input_channel, self.output_size**2,
                              kernel_size=1, stride=1, padding=0)

    def forward(self, input: Tensor):
        output = self.conv(input)
        return output


class GeometricTerm(nn.Module):
    def __init__(self):
        super(GeometricTerm, self).__init__()
        self.boxTerm = BoxTerm()
        self.positionTerm = PositionTerm()

    def forward(self, input: Tensor, rois: Tensor):
        box_output = self.boxTerm(rois)
        position_output = self.positionTerm([i for i in input.shape])
        geometric_output = []
        for i, img in enumerate(position_output):
            i = torch.tensor(i, dtype=torch.int64)
            for box in box_output:
                if box[0] == i:
                    temp = [i]
                    temp.append(torch.cat(
                        [torch.matmul(vector.reshape(1, -1), img).squeeze().unsqueeze(0)
                         for vector in box[1:]]
                    ))
                    geometric_output.append(temp)
        return geometric_output


class LearnableRoIPool(nn.Module):
    """
    See roi_pool
    """

    def __init__(self, input_channel: int, output_size: BroadcastingList2[int], spatial_scale: float):
        super(LearnableRoIPool, self).__init__()
        self.input_channel = input_channel
        self.output_size = output_size
        self.spatial_scale = spatial_scale

        self.geometricTerm = GeometricTerm()
        self.appearanceTerm = AppearanceTerm(self.input_channel)
        self.softmax=nn.Softmax2d()

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        geo = self.geometricTerm(input, rois)
        app = self.appearanceTerm(input)
        print(app.shape)
        
        weights=[]
        for i in range(input.shape[0]):
            i = torch.tensor(i, dtype=torch.int64)           
            for box in geo:
                if box[0] == i:
                    temp = [i]
                    temp.append(self.softmax(app + box[1]))
                    weights.append(temp)
        
        return weights

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ')'
        return tmpstr


rois = torch.tensor([[0, 2.1, 2.3, 3.5, 3.8],
                     [1, 2.2, 2.4, 3.5, 3.9]])
# print(box_embedding(rois))

# boxTerm = BoxTerm()
# print(boxTerm(rois)[0][1].shape)

# input_size = [5, 4, 30, 20]
# print(positions_embedding(input_size)[0].shape)

# positionTerm = PositionTerm()
# print(positionTerm(input_size)[0].shape)

# geometricTerm = GeometricTerm()
input = torch.randn(5, 4, 30, 20)
# print(geometricTerm(input, rois)[0][1].shape)
# print(len(geometricTerm(input, rois)))

learnableRoIPool=LearnableRoIPool(4, 7, 1.0)
print(learnableRoIPool(input, rois)[0][1].shape)
