import os
import requests
from requests.adapters import HTTPAdapter

import torch
from torch import nn
from torch.nn import functional as F

from utils.download import download_url_to_file

import escnn.gspaces as gspaces
import escnn.nn as enn


class BasicEquivariantConv2d(enn.EquivariantModule):
    def __init__(self, in_type, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        gspace = in_type.gspace

        self.out_type = enn.FieldType(gspace, out_channels * [gspace.regular_repr])

        self.conv = enn.R2Conv(
            in_type,
            self.out_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = enn.InnerBatchNorm(self.out_type)
        self.relu = enn.ReLU(self.out_type, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def evaluate_output_shape(self, input_shape):
        return self.out_type.size, input_shape[1], input_shape[2]



class EquivariantBlock35(enn.EquivariantModule):
    def __init__(self, in_type, scale=0.17):
        super().__init__()

        self.in_type = in_type
        self.scale = scale
        gspace = in_type.gspace

        self.branch0 = BasicEquivariantConv2d(in_type, 32, kernel_size=1, stride=1)

        self.branch1a = BasicEquivariantConv2d(in_type, 32, kernel_size=1, stride=1)
        self.branch1b = BasicEquivariantConv2d(self.branch1a.out_type, 32, kernel_size=3, stride=1, padding=1)

        self.branch2a = BasicEquivariantConv2d(in_type, 32, kernel_size=1, stride=1)
        self.branch2b = BasicEquivariantConv2d(self.branch2a.out_type, 32, kernel_size=3, stride=1, padding=1)
        self.branch2c = BasicEquivariantConv2d(self.branch2b.out_type, 32, kernel_size=3, stride=1, padding=1)

        concat_type = enn.FieldType(gspace, 96 * [gspace.regular_repr])
        self.out_type = in_type

        self.conv_linear = enn.R2Conv(
            concat_type,
            self.out_type,
            kernel_size=1,
            bias=False
        )

        self.relu = enn.ReLU(self.out_type, inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)

        x1 = self.branch1a(x)
        x1 = self.branch1b(x1)

        x2 = self.branch2a(x)
        x2 = self.branch2b(x2)
        x2 = self.branch2c(x2)

        out = enn.tensor_directsum([x0, x1, x2])

        out = self.conv_linear(out)

        out = out * self.scale + x

        out = self.relu(out)
        return out

    def evaluate_output_shape(self, input_shape):
        return self.out_type.size, input_shape[1], input_shape[2]



class EquivariantBlock17(enn.EquivariantModule):
    def __init__(self, in_type, scale=0.10):
        super().__init__()

        self.in_type = in_type
        self.scale = scale
        gspace = in_type.gspace

        # Branch 0: 1x1 conv
        self.branch0 = BasicEquivariantConv2d(in_type, 128, kernel_size=1, stride=1)

        # Branch 1: 1x1 → (1x7) → (7x1) convolutions
        self.branch1a = BasicEquivariantConv2d(in_type, 128, kernel_size=1, stride=1)
        self.branch1b = BasicEquivariantConv2d(self.branch1a.out_type, 128, kernel_size=3, stride=1, padding=1)
        self.branch1c = BasicEquivariantConv2d(self.branch1b.out_type, 128, kernel_size=3, stride=1, padding=1)

        # Concatenation output: 128 + 128 = 256
        concat_type = enn.FieldType(gspace, 256 * [gspace.regular_repr])

        # Final 1x1 to project back to input channels
        self.out_type = in_type
        self.conv_linear = enn.R2Conv(concat_type, self.out_type, kernel_size=1, bias=False)

        self.relu = enn.ReLU(self.out_type, inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)

        x1 = self.branch1a(x)
        x1 = self.branch1b(x1)
        x1 = self.branch1c(x1)

        out = enn.tensor_directsum([x0, x1])
        out = self.conv_linear(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

    def evaluate_output_shape(self, input_shape):
        return self.out_type.size, input_shape[1], input_shape[2]


class EquivariantBlock8(enn.EquivariantModule):
    def __init__(self, in_type, scale=0.20, noReLU=False):
        super().__init__()

        self.in_type = in_type
        self.scale = scale
        self.noReLU = noReLU
        gspace = in_type.gspace

        # Branch 0: 1x1 conv
        self.branch0 = BasicEquivariantConv2d(in_type, 192, kernel_size=1, stride=1)

        # Branch 1: 1x1 → (1x3) → (3x1) convolutions
        self.branch1a = BasicEquivariantConv2d(in_type, 192, kernel_size=1, stride=1)
        self.branch1b = BasicEquivariantConv2d(self.branch1a.out_type, 192, kernel_size=3, stride=1, padding=1)
        self.branch1c = BasicEquivariantConv2d(self.branch1b.out_type, 192, kernel_size=3, stride=1, padding=1)

        # Concatenation output: 192 + 192 = 384
        concat_type = enn.FieldType(gspace, 384 * [gspace.regular_repr])

        # Final 1x1 to project back to input channels
        self.out_type = in_type
        self.conv_linear = enn.R2Conv(concat_type, self.out_type, kernel_size=1, bias=False)

        if not self.noReLU:
            self.relu = enn.ReLU(self.out_type, inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)

        x1 = self.branch1a(x)
        x1 = self.branch1b(x1)
        x1 = self.branch1c(x1)

        out = enn.tensor_directsum([x0, x1])
        out = self.conv_linear(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out

    def evaluate_output_shape(self, input_shape):
        return self.out_type.size, input_shape[1], input_shape[2]

class EquivariantMixed_6a(enn.EquivariantModule):
    def __init__(self, in_type):
        super().__init__()

        self.in_type = in_type
        gspace = in_type.gspace

        # Branch 0: 3x3 conv with stride=2 (downsampling)
        self.branch0 = BasicEquivariantConv2d(in_type, 384, kernel_size=3, stride=2)

        # Branch 1: 1x1 → 3x3 → 3x3 (last one downsamples)
        self.branch1a = BasicEquivariantConv2d(in_type, 192, kernel_size=1, stride=1)
        self.branch1b = BasicEquivariantConv2d(self.branch1a.out_type, 192, kernel_size=3, stride=1, padding=1)
        self.branch1c = BasicEquivariantConv2d(self.branch1b.out_type, 256, kernel_size=3, stride=2)

        # Branch 2: Max pooling (must use PointwiseMaxPool)
        self.branch2 = enn.PointwiseMaxPool(in_type, kernel_size=3, stride=2)

        # Output field type: 384 + 256 + in_type.size (from pooling)
        self.out_type = enn.FieldType(gspace, 
            (self.branch0.out_type.size + self.branch1c.out_type.size + self.branch2.out_type.size) * [gspace.regular_repr]
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1 = self.branch1a(x)
        x1 = self.branch1b(x1)
        x1 = self.branch1c(x1)

        x2 = self.branch2(x)

        out = enn.tensor_directsum([x0, x1, x2])
        return out

    def evaluate_output_shape(self, input_shape):
        return self.out_type.size, input_shape[1], input_shape[2]

class EquivariantMixed_7a(enn.EquivariantModule):
    def __init__(self, in_type):
        super().__init__()

        self.in_type = in_type
        gspace = in_type.gspace

        # Branch 0: 1x1 → 3x3 (stride=2)
        self.branch0a = BasicEquivariantConv2d(in_type, 256, kernel_size=1, stride=1)
        self.branch0b = BasicEquivariantConv2d(self.branch0a.out_type, 384, kernel_size=3, stride=2)

        # Branch 1: 1x1 → 3x3 (stride=2)
        self.branch1a = BasicEquivariantConv2d(in_type, 256, kernel_size=1, stride=1)
        self.branch1b = BasicEquivariantConv2d(self.branch1a.out_type, 256, kernel_size=3, stride=2)

        # Branch 2: 1x1 → 3x3 → 3x3 (last one stride=2)
        self.branch2a = BasicEquivariantConv2d(in_type, 256, kernel_size=1, stride=1)
        self.branch2b = BasicEquivariantConv2d(self.branch2a.out_type, 256, kernel_size=3, stride=1, padding=1)
        self.branch2c = BasicEquivariantConv2d(self.branch2b.out_type, 256, kernel_size=3, stride=2)

        # Branch 3: Max pool (stride=2)
        self.branch3 = enn.PointwiseMaxPool(in_type, kernel_size=3, stride=2)

        # Compute output FieldType
        out_size = (
            self.branch0b.out_type.size +
            self.branch1b.out_type.size +
            self.branch2c.out_type.size +
            self.branch3.out_type.size
        )
        self.out_type = enn.FieldType(gspace, out_size * [gspace.regular_repr])

    def forward(self, x):
        x0 = self.branch0a(x)
        x0 = self.branch0b(x0)

        x1 = self.branch1a(x)
        x1 = self.branch1b(x1)

        x2 = self.branch2a(x)
        x2 = self.branch2b(x2)
        x2 = self.branch2c(x2)

        x3 = self.branch3(x)

        out = enn.tensor_directsum([x0, x1, x2, x3])
        return out

    def evaluate_output_shape(self, input_shape):
        return self.out_type.size, input_shape[1], input_shape[2]

class EquivariantInceptionResnetV1(nn.Module):
    """Inception Resnet V1 model with optional loading of pretrained weights.

    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.

    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """
    def __init__(self, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None):
        super().__init__()

        self.r2_act = gspaces.rot2dOnR2(N=8)  


        # Set simple attributes
        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes

        if pretrained == 'vggface2':
            tmp_classes = 8631
        elif pretrained == 'casia-webface':
            tmp_classes = 10575
        elif pretrained is None and self.classify and self.num_classes is None:
            raise Exception('If "pretrained" is not specified and "classify" is True, "num_classes" must be specified')


        # Define layers
        self.input_type = enn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr]) # For input RGB images


        print("1")
        self.conv2d_1a = BasicEquivariantConv2d(self.input_type, 32, kernel_size=3, stride=2)
        print("2")
        self.conv2d_2a = BasicEquivariantConv2d(self.conv2d_1a.out_type, 32, kernel_size=3, stride=1)
        print("3")
        self.conv2d_2b = BasicEquivariantConv2d(self.conv2d_2a.out_type, 64, kernel_size=3, stride=1, padding=1)
        print("4")
        self.maxpool_3a = enn.PointwiseMaxPool(self.conv2d_2b.out_type, kernel_size=3, stride=2)
        print("5")
        self.conv2d_3b = BasicEquivariantConv2d(self.maxpool_3a.out_type, 80, kernel_size=1, stride=1)
        print("6")
        self.conv2d_4a = BasicEquivariantConv2d(self.conv2d_3b.out_type, 192, kernel_size=3, stride=1)
        print("7")
        self.conv2d_4b = BasicEquivariantConv2d(self.conv2d_4a.out_type, 256, kernel_size=3, stride=2)
        print("8")
        self.block35_1 = EquivariantBlock35(self.conv2d_4b.out_type, scale=0.17)
        print("9")
        self.block35_2 = EquivariantBlock35(self.block35_1.out_type, scale=0.17)
        self.block35_3 = EquivariantBlock35(self.block35_2.out_type, scale=0.17)
        self.block35_4 = EquivariantBlock35(self.block35_3.out_type, scale=0.17)
        self.block35_5 = EquivariantBlock35(self.block35_4.out_type, scale=0.17)
        # After block35_5
        self.mixed_6a = EquivariantMixed_6a(self.block35_5.out_type)

        # Build repeat_2: 10 EquivariantBlock17s
        self.block17_1 = EquivariantBlock17(self.mixed_6a.out_type, scale=0.10)
        self.block17_2 = EquivariantBlock17(self.block17_1.out_type, scale=0.10)
        self.block17_3 = EquivariantBlock17(self.block17_2.out_type, scale=0.10)
        self.block17_4 = EquivariantBlock17(self.block17_3.out_type, scale=0.10)
        self.block17_5 = EquivariantBlock17(self.block17_4.out_type, scale=0.10)
        self.block17_6 = EquivariantBlock17(self.block17_5.out_type, scale=0.10)
        self.block17_7 = EquivariantBlock17(self.block17_6.out_type, scale=0.10)
        self.block17_8 = EquivariantBlock17(self.block17_7.out_type, scale=0.10)
        self.block17_9 = EquivariantBlock17(self.block17_8.out_type, scale=0.10)
        self.block17_10 = EquivariantBlock17(self.block17_9.out_type, scale=0.10)

        # Mixed 7a after block17
        self.mixed_7a = EquivariantMixed_7a(self.block17_10.out_type)

        # Build repeat_3: 5 EquivariantBlock8s
        self.block8_1 = EquivariantBlock8(self.mixed_7a.out_type, scale=0.20)
        self.block8_2 = EquivariantBlock8(self.block8_1.out_type, scale=0.20)
        self.block8_3 = EquivariantBlock8(self.block8_2.out_type, scale=0.20)
        self.block8_4 = EquivariantBlock8(self.block8_3.out_type, scale=0.20)
        self.block8_5 = EquivariantBlock8(self.block8_4.out_type, scale=0.20)

        # Final block8 without ReLU
        self.block8_final = EquivariantBlock8(self.block8_5.out_type, scale=1.0, noReLU=True)

        # After that — avgpool and dense layers
        self.final_feature_type = self.block8_final.out_type

        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(self.final_feature_type.size, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

        if pretrained is not None:
            self.logits = nn.Linear(512, tmp_classes)
            load_weights(self, pretrained)

        if self.classify and self.num_classes is not None:
            self.logits = nn.Linear(512, self.num_classes)

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, x):
        """Calculate embeddings or logits given a batch of input image tensors.

        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.

        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)

        # Block35 repeat
        x = self.block35_1(x)
        x = self.block35_2(x)
        x = self.block35_3(x)
        x = self.block35_4(x)
        x = self.block35_5(x)

        # Mixed6a
        x = self.mixed_6a(x)

        # Block17 repeat
        x = self.block17_1(x)
        x = self.block17_2(x)
        x = self.block17_3(x)
        x = self.block17_4(x)
        x = self.block17_5(x)
        x = self.block17_6(x)
        x = self.block17_7(x)
        x = self.block17_8(x)
        x = self.block17_9(x)
        x = self.block17_10(x)

        # Mixed7a
        x = self.mixed_7a(x)

        # Block8 repeat
        x = self.block8_1(x)
        x = self.block8_2(x)
        x = self.block8_3(x)
        x = self.block8_4(x)
        x = self.block8_5(x)

        # Final Block8
        x = self.block8_final(x)

        # Pooling and projection
        x = self.avgpool_1a(x.tensor)  # Important: use x.tensor because avgpool is not equivariant
        x = self.dropout(x)
        x = self.last_linear(x.view(x.size(0), -1))
        x = self.last_bn(x)

        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x



def load_weights(mdl, name):
    """Download pretrained state_dict and load into model.

    Arguments:
        mdl {torch.nn.Module} -- Pytorch model.
        name {str} -- Name of dataset that was used to generate pretrained state_dict.

    Raises:
        ValueError: If 'pretrained' not equal to 'vggface2' or 'casia-webface'.
    """
    if name == 'vggface2':
        path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt'
    elif name == 'casia-webface':
        path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt'
    else:
        raise ValueError('Pretrained models only exist for "vggface2" and "casia-webface"')

    model_dir = os.path.join(get_torch_home(), 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)

    cached_file = os.path.join(model_dir, os.path.basename(path))
    if not os.path.exists(cached_file):
        download_url_to_file(path, cached_file)

    state_dict = torch.load(cached_file)
    mdl.load_state_dict(state_dict)


def get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            'TORCH_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')
        )
    )
    return torch_home