from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import nms

#------------------------------------------------------------------------#
#   YOLOv3 输出解码实现
#------------------------------------------------------------------------#

class DecodeBox(nn.Module):
    """ 预测结果的解码 """
    def __init__(self, anchors, num_classes, img_size):
        """ 初始化参数：先验框，类别数，图像尺寸 """
        super(DecodeBox, self).__init__()
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        #-----------------------------------------------------------#
        self.anchors = anchors
        self.num_anchors = len(anchors)                 # 先验框的数目
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes               # 每个 Box：x, y, h, w, confidence, classes, sum=5+3=8
        self.img_size = img_size
    # end

    def forward(self, input):
        """ 返回解码后的结果（尚未进行筛选） """
        #-----------------------------------------------#
        #   输入的 input可能有三种，他们的 shape分别是
        #   B           C    H   W
        #   batch_size, 24, 13, 13
        #   batch_size, 24, 26, 26
        #   batch_size, 24, 52, 52
        #-----------------------------------------------#
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        #-----------------------------------------------#
        #   将整个图像用网格分割后，单个grid cell的像素宽高
        #   当 img_size为 416x416时：
        #   stride_h = stride_w = 32、16、8（结果因input而异）
        #-----------------------------------------------#
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width

        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #   即：anchor的宽高相当于多少个网格（以grid cell为单位）
        #-------------------------------------------------#
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors]

        #-----------------------------------------------#
        #   输入的input可能有三种，将他们从原尺寸reshape为：
        #   B           Anchors   H   W  Attrs
        #   batch_size, 3,       13, 13,    8
        #   batch_size, 3,       26, 26,    8
        #   batch_size, 3,       52, 52,    8
        #   C = Anchors* Attrs = 24
        #-----------------------------------------------#
        prediction = input.view(batch_size, self.num_anchors,          # 注意原始 input维度分布情况，先局部拆分，再交换位置，保持连续性
                                self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        # 下面从prediction的最后一个维度提取并加工预测单项
        # 先验框的中心位置的调整参数 (Sigmoid, 因 x,y在grid cell内，x,y∈(0, 1))
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])

        # 先验框的宽高调整参数（可能大于1）
        w = prediction[..., 2]
        h = prediction[..., 3]

        # 获得是否有物体的置信度 (Sigmoid，二元分类)
        conf = torch.sigmoid(prediction[..., 4])

        # 获得种类置信度 (Sigmoid，也看做二元分类)（因为是所有的类别，即该维度剩余的所有数据）
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # 获取设备端 or 主机端 Tensor 生成器，简化代码书写
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        #----------------------------------------------------------#
        #   由于每个gridcell预测结果中的 x,y 值是在 0~1区间内的；
        #   所以原始预测结果需要加上对应网格的index，
        #   才是最终在整幅图像上的位置（以单位网格计算）
        #   下面的代码分别生成了如下尺寸的 x,y index表（H、W因输入异）：
        #   B           Anchors     H   W
        #   batch_size, 3,         13, 13
        #   batch_size, 3,         26, 26
        #   batch_size, 3,         52, 52
        #----------------------------------------------------------#
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        #----------------------------------------------------------#
        #   由于每个gridcell预测结果中的 w,h 实际上是 anchor的比例系数；
        #   所以原始预测值需要乘以对应的 anchor的 w,h ，
        #   才是最终的预测结果（以单位网格计算）
        #   下面的代码分别生成了如下尺寸的 anchor宽高表（H、W因输入异）：
        #   B           Anchors     H   W
        #   batch_size, 3,         13, 13
        #   batch_size, 3,         26, 26
        #   batch_size, 3,         52, 52
        #----------------------------------------------------------#
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))             # 将 W、H分别摘出来
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)    # ？先把 anchor_w竖起来再构造更好
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        #----------------------------------------------------------#
        #   利用原始预测结果，结合上面的准备数据，得出最终预测结果
        #   1. 调整先验框的中心，从先验框中心向右下角偏移
        #   2. 调整先验框的宽高
        #   这么做的原因：降低网络拟合的难度，利于收敛
        #----------------------------------------------------------#
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w          # exp: > 0
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        #-----------------------------------------------------------------------#
        #   将输出结果调整成相对于输入图像大小（以像素为单位，不再以gridcell为单位）
        #-----------------------------------------------------------------------#
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)       # 注意：这里不是数乘。对应：pred_boxes[..., 0/1/2/3]
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,        # 这里把 H、W维度合起来了，因为不再需要二维结构
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data
    # end
# end

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)/input_shape
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)/input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
                 
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    #----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角的格式。
    #   prediction  [batch_size, num_anchors, 8]
    #----------------------------------------------------------#
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        #----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        #----------------------------------------------------------#
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        #----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        #----------------------------------------------------------#
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        #----------------------------------------------------------#
        #   根据置信度进行预测结果的筛选
        #----------------------------------------------------------#
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        #-------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        #-------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        #------------------------------------------#
        #   获得预测结果中包含的所有种类
        #------------------------------------------#
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            #------------------------------------------#
            #   获得某一类得分筛选后全部的预测结果
            #------------------------------------------#
            detections_class = detections[detections[:, -1] == c]

            #------------------------------------------#
            #   使用官方自带的非极大抑制会速度更快一些！
            #------------------------------------------#
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4] * detections_class[:, 5],
                nms_thres
            )
            max_detections = detections_class[keep]
            
            # # 按照存在物体的置信度排序
            # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
            # detections_class = detections_class[conf_sort_index]
            # # 进行非极大抑制
            # max_detections = []
            # while detections_class.size(0):
            #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
            #     max_detections.append(detections_class[0].unsqueeze(0))
            #     if len(detections_class) == 1:
            #         break
            #     ious = bbox_iou(max_detections[-1], detections_class[1:])
            #     detections_class = detections_class[1:][ious < nms_thres]
            # # 堆叠
            # max_detections = torch.cat(max_detections).data
            
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output
