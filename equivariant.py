import os
import requests

import torch
from torch import nn
from torch.nn import functional as F

from utils.download import download_url_to_file

import escnn.gspaces as gspaces
import escnn.nn as enn


class BasicEquivariantConv2d(enn.EquivariantModule):
    def __init__(self,
                 in_type: enn.FieldType,
                 out_fields: int,
                 kernel_size,
                 stride: int = 1,
                 padding: int = 0):
        super().__init__()
        gspace = in_type.gspace 

        self.out_type = enn.FieldType(
            gspace,
            [gspace.trivial_repr] * out_fields
        )

        self.conv = enn.R2Conv(
            in_type,
            self.out_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        self.bn   = enn.InnerBatchNorm(
            self.out_type, eps=1e-5, momentum=0.1, affine=True
        )
        self.relu = enn.ReLU(self.out_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def evaluate_output_shape(self, input_shape):
        h_in, w_in = input_shape[1], input_shape[2]
        h_out = (h_in + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1) // self.conv.stride[0] + 1
        w_out = (w_in + 2 * self.conv.padding[1] - self.conv.dilation[1] * (self.conv.kernel_size[1] - 1) - 1) // self.conv.stride[1] + 1
        return (self.out_type.size, h_out, w_out)


class EquivariantInceptionBlock(enn.EquivariantModule):
    def __init__(self, in_type: enn.FieldType, out_fields1: int, out_fields3: int):
        super().__init__()
        self.branch1 = BasicEquivariantConv2d(
            in_type=in_type,
            out_fields=out_fields1,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.branch2 = BasicEquivariantConv2d(
            in_type=in_type,
            out_fields=out_fields3,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.out_type = enn.FieldType(
            in_type.gspace,
            self.branch1.out_type.representations + self.branch2.out_type.representations
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return enn.tensor_directsum([x1, x2])
        
    def evaluate_output_shape(self, input_shape):
        return (self.out_type.size, input_shape[1], input_shape[2])


class EquivariantDownsampleBlock(enn.EquivariantModule):
    def __init__(self, in_type: enn.FieldType, out_fields3: int):
        super().__init__()
        self.branch1 = BasicEquivariantConv2d(
            in_type=in_type,
            out_fields=out_fields3,
            kernel_size=3,
            stride=2,
            padding=0
        )

        self.branch2 = enn.PointwiseMaxPool(
            in_type, kernel_size=3, stride=2, padding=0
        )

        self.out_type = enn.FieldType(
            in_type.gspace,
            self.branch1.out_type.representations
            + self.branch2.out_type.representations
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        return enn.tensor_directsum([x1, x2])
        
    def evaluate_output_shape(self, input_shape):
        shape1 = self.branch1.evaluate_output_shape(input_shape)
        shape2 = self.branch2.evaluate_output_shape(input_shape)

        assert shape1[1] == shape2[1] and shape1[2] == shape2[2], \
            f"Spatial dimensions mismatch after downsampling: {shape1} vs {shape2}"

        total_fields = self.out_type.size
        return (total_fields, shape1[1], shape1[2])


class EquivariantSmallInception(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.gspace = gspaces.rot2dOnR2(N=-1)
        self.input_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)
        
        # Update out_fields to match baseline channel counts directly
        self.conv1 = BasicEquivariantConv2d(
            in_type=self.input_type,
            out_fields=96, # Baseline: 96 channels
            kernel_size=3, stride=1, padding=0
        )
        
        self.inception1a = EquivariantInceptionBlock(
            in_type=self.conv1.out_type, # 96 fields in
            out_fields1=32, # Baseline: 32
            out_fields3=32  # Baseline: 32 -> 64 fields total
        )
        
        self.inception1b = EquivariantInceptionBlock(
            in_type=self.inception1a.out_type, # 64 fields in
            out_fields1=32, # Baseline: 32
            out_fields3=48  # Baseline: 48 -> 80 fields total
        )
        
        self.downsample1 = EquivariantDownsampleBlock(
            in_type=self.inception1b.out_type, # 80 fields in
            out_fields3=80 # Baseline: 80. Pool branch keeps 80 fields. -> 160 fields total
        )
        
        self.inception2a = EquivariantInceptionBlock(
            in_type=self.downsample1.out_type, # 160 fields in
            out_fields1=112, # Baseline: 112
            out_fields3=48   # Baseline: 48 -> 160 fields total
        )
        
        self.inception2b = EquivariantInceptionBlock(
            in_type=self.inception2a.out_type, # 160 fields in
            out_fields1=96, # Baseline: 96
            out_fields3=64  # Baseline: 64 -> 160 fields total
        )
        
        self.inception2c = EquivariantInceptionBlock(
            in_type=self.inception2b.out_type, # 160 fields in
            out_fields1=80, # Baseline: 80
            out_fields3=80  # Baseline: 80 -> 160 fields total
        )
        
        self.inception2d = EquivariantInceptionBlock(
            in_type=self.inception2c.out_type, # 160 fields in
            out_fields1=48, # Baseline: 48
            out_fields3=96  # Baseline: 96 -> 144 fields total
        )
        
        self.downsample2 = EquivariantDownsampleBlock(
            in_type=self.inception2d.out_type, # 144 fields in
            out_fields3=96 # Baseline: 96. Pool branch keeps 144 fields. -> 240 fields total
        )
        
        self.inception3a = EquivariantInceptionBlock(
            in_type=self.downsample2.out_type, # 240 fields in
            out_fields1=176, # Baseline: 176
            out_fields3=160  # Baseline: 160 -> 336 fields total
        )
        
        self.inception3b = EquivariantInceptionBlock(
            in_type=self.inception3a.out_type, # 336 fields in
            out_fields1=176, # Baseline: 176
            out_fields3=160  # Baseline: 160 -> 336 fields total
        )
        
        final_type = self.inception3b.out_type

        num_invariant_features = final_type.size
        self.fully_connected = nn.Linear(num_invariant_features, num_classes)
        self.dropout = nn.Dropout(0)


    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        x = self.conv1(x)
        x = self.inception1a(x)
        x = self.inception1b(x)
        x = self.downsample1(x)
        x = self.inception2a(x)
        x = self.inception2b(x)
        x = self.inception2c(x)
        x = self.inception2d(x)
        x = self.downsample2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = x.tensor 

        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)

        x = self.fully_connected(x)
        x = self.dropout(x)
        return x
