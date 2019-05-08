"""
MIT License

Copyright (c) 2019 Microsoft

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class HRFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 normalize=None,
                 pooling='AVG',
                 share_conv=False,
                 conv_stride=1,
                 with_checkpoint=False):
        super(HRFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.with_bias = normalize is None
        self.share_conv = share_conv
        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=sum(in_channels),
                      out_channels=out_channels,
                      kernel_size=1),
        )

        if self.share_conv:
            self.fpn_conv = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3, 
                stride=conv_stride,
                padding=1,
            )
        else:
            self.fpn_conv = nn.ModuleList()
            for i in range(5):
                self.fpn_conv.append(nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=conv_stride,
                    padding=1
                ))
        if pooling == 'MAX':
            print("Using AVG Pooling")
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d
        self.with_checkpoint = with_checkpoint

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,  a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outs = []
        outs.append(inputs[0])
        for i in range(1, len(inputs)):
            outs.append(F.interpolate(inputs[i], scale_factor=2**i, mode='bilinear'))
        out = torch.cat(outs, dim=1)
        if out.requires_grad and self.with_checkpoint:
            out = checkpoint(self.reduction_conv, out)
        else:
            out = self.reduction_conv(out)
        outs = [out]
        for i in range(1, 5):
            outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))
        outputs = []
        if self.share_conv:
            for i in range(5):
                outputs.append(self.fpn_conv(outs[i]))
        else:
            for i in range(5):
                if outs[i].requires_grad and self.with_checkpoint:
                    tmp_out = checkpoint(self.fpn_conv[i], outs[i])
                else:
                    tmp_out = self.fpn_conv[i](outs[i])
                outputs.append(tmp_out)
        return tuple(outputs)


class HRFPNv2(nn.Module):

    layer_config2 = [
        [["c2"], ["c2_", "_c2"], ["p2", "c2_", "_c2"], ["p4", "c2_", "_c2"], ["p8", "c2_", "_c2"]],
        [["c1"], ["c2"], ["p2", "c2"], ["p2", "c2_", "_c2"], ["p4", "c2_", "_c2"]],
        [["d2", "c1"], ["c1"], ["c2"], ["p2", "c2"], ["p2", "c2_", "_c2"]],
        [["d2", "c1_", "d2", "_c1"], ["d2", "c1"], ["c1"], ["c2"], ["c2", "c2"]]
    ]
    layer_config1 = [
        [["c1"], ["c2"], ["c2_", "_c2"], ["p2", "c2_","_c2"], ["p4", "c2_", "_c2"]],
        [["d2", "c1"], ["c1"], ["c2"], ["c2_", "_c2"], ["p2", "c2_","_c2"]],
        [["d2", "c1_", "d2", "_c1"], ["d2", "c1"], ["c1"], ["c2"], ["p2", "c2"]],
        [["d4", "c1_", "d2", "_c1"], ["d2", "c1_", "d2", "_c1"], ["d2", "c1"], ["c1"], ["c2"]],
    ]

    def __init__(self, in_channels, out_channels, conv_stride=2):
        super(HRFPNv2, self).__init__()
        assert isinstance(in_channels, list)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_branches = 5

        self.fuse_layer = self._make_fuse_layers(
            self.layer_config2 if conv_stride == 2 else self.layer_config1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,  a=1)
    
    def _make_fuse_layers(self, layer_config):
        in_channels = self.in_channels
        out_channels = self.out_channels

        fuse_layers = []
        for i in range(len(in_channels)):
            fuse_layer_per_in = []
            for j in range(self.num_branches):
                fuse_layer_per_inout = []
                for item in layer_config[i][j]:
                    if item == "c2":
                        fuse_layer_per_inout += [
                            nn.Conv2d(in_channels[i], out_channels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(out_channels, momentum=0.1)]
                    elif item == "c2_":
                        fuse_layer_per_inout += [
                            nn.Conv2d(in_channels[i], out_channels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(out_channels, momentum=0.1),
                            nn.ReLU(False)]
                    elif item == "_c2":
                        fuse_layer_per_inout += [
                            nn.Conv2d(out_channels, out_channels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(out_channels, momentum=0.1)]
                    elif item == "c1":
                        fuse_layer_per_inout += [
                            nn.Conv2d(in_channels[i], out_channels, 3, 1, 1, bias=False),
                            nn.BatchNorm2d(out_channels, momentum=0.1)]
                    elif item == "c1_":
                        fuse_layer_per_inout += [
                            nn.Conv2d(in_channels[i], out_channels, 3, 1, 1, bias=False),
                            nn.BatchNorm2d(out_channels, momentum=0.1),
                            nn.ReLU(False)]
                    elif item == "_c1":
                        fuse_layer_per_inout += [
                            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                            nn.BatchNorm2d(out_channels, momentum=0.1)]
                    elif item == "p2":
                        fuse_layer_per_inout += [nn.AvgPool2d(2, 2, 0)]
                    elif item == "p4":
                        fuse_layer_per_inout += [nn.AvgPool2d(4, 4, 0)]
                    elif item == "p8":
                        fuse_layer_per_inout += [nn.AvgPool2d(8, 8, 0)]
                    elif item == "d2":
                        fuse_layer_per_inout += [nn.Upsample(scale_factor=2, mode='nearest')]
                    elif item == "d4":
                        fuse_layer_per_inout += [nn.Upsample(scale_factor=4, mode='nearest')]
                    else:
                        raise NotImplementedError
                fuse_layer_per_in.append(nn.Sequential(*fuse_layer_per_inout))
            fuse_layers.append(nn.ModuleList(fuse_layer_per_in))
        return nn.ModuleList(fuse_layers)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        outs = []
        for j in range(self.num_branches):
            out = []
            for i in range(len(self.in_channels)):
                out.append(self.fuse_layer[i][j](inputs[i]))
            outs.append(out)
        
        outs = [torch.stack(out, dim=0).sum(dim=0) for out in outs]

        return tuple(outs)
