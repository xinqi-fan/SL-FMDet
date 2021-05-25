from collections import OrderedDict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torchvision.models as models

from models.net import MobileNetV1
from models.net import FPN, SSH, RCAM


# import resnet

def conv_bn1X1_out(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.Sigmoid()
    )


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3, num_class=3):
        super(ClassHead, self).__init__()
        self.num_class = num_class
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * self.num_class, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, self.num_class)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class FaceMaskDetector(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(FaceMaskDetector, self).__init__()
        self.phase = phase
        backbone = None
        self.cfg = cfg
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_imagenet_pretrain.tar", map_location=torch.device('cpu'))
                # remove prefix module, which trained on multiple GPU
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)            

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']

        self.fpn = FPN(in_channels_list, out_channels)

        if cfg['attention']:
            self.context1 = RCAM(out_channels, out_channels)
            self.context2 = RCAM(out_channels, out_channels)
            self.context3 = RCAM(out_channels, out_channels)
        else:
            self.context1 = SSH(out_channels, out_channels)
            self.context2 = SSH(out_channels, out_channels)
            self.context3 = SSH(out_channels, out_channels)

        self.feature2heatmap = conv_bn1X1_out(out_channels, 1, stride=1)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'], num_classes=cfg['num_classes'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2, num_classes=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num, num_classes))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # Context
        feature1 = self.context1(fpn[0])
        feature2 = self.context2(fpn[1])
        feature3 = self.context3(fpn[2])
        features = [feature1, feature2, feature3]

        # heatmap mapping
        heatmap_correspond = self.feature2heatmap(feature2)

        # detection head
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1))
        return output, heatmap_correspond

