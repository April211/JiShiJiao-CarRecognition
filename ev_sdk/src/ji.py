from __future__ import print_function

import logging as log
import os
import pathlib
import json
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import colorsys
import time

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from utils import (DecodeBox, letterbox_image, non_max_suppression,
                         yolo_correct_boxes)

# 测试代码

device = 'CPU'
input_h, input_w, input_c, input_n = (416, 416, 3, 1)

def init():
    """
    Initialize model
    Returns: model
    """
    model_xml = "/project/ev_sdk/model/Yolov3.xml"
    if not os.path.isfile(model_xml):
        log.error(f'{model_xml} does not exist!!!')
        return None

    model_bin = pathlib.Path(model_xml).with_suffix('.bin').as_posix()
    net = IENetwork(model=model_xml, weights=model_bin)

    # Load Inference Engine
    ie = IECore()
    global exec_net
    exec_net = ie.load_network(network=net, device_name=device)

    input_blob = next(iter(net.inputs))
    n, c, h, w = net.inputs[input_blob].shape
    global input_h, input_w, input_c, input_n
    input_h, input_w, input_c, input_n = h, w, c, n

    return net
# end

#---------------------------------------------------#
#   获得所有的先验框
#---------------------------------------------------#
def _get_anchors():
    anchors_path = os.path.expanduser("/project/ev_sdk/src/yolo_anchors.txt")
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1,:,:]


def process_image(net=None, input_image=None, args=None, **kwargs):
    """
    Do inference to analysis input_image and get output
    Attributes:
        net: algorithm handle returned by init()
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
    Returns: process result
    """

    image_shape = np.array(np.shape(input_image)[0:2])

    # Process image here
    image = Image.fromarray(np.uint8(input_image))              # PIL
    b, g, r = image.split()
    image = Image.merge("RGB", (r, g, b))
    image = image.convert('RGB')

    crop_img = image.resize((input_h, input_w), Image.BICUBIC)      # hwc
    photo = np.array(crop_img, dtype = np.float32) / 255.0
    photo = np.transpose(photo, (2, 0, 1))              # hwc -> chw
    images = np.ndarray(shape=(input_n, input_c, input_h, input_w))
    images[0] = photo                                  # numpy, 4D

    # iters
    input_blob = next(iter(net.inputs))
    iterator = iter(net.outputs)
    out_blob = next(iterator)


    num_classes = 3
    confidence = 0.5
    model_image_size = (416, 416, 3)
    class_names = ['car', 'tricar', 'motorbike']
    iou = 0.3

    # --------------------------- Performing inference ----------------------------------------------------
    anchors = _get_anchors()                    # 获取先验框
    yolo_decodes = []                           # 解码器列表
    for i in range(3):
        yolo_decodes.append(DecodeBox(anchors[i], 3, (input_h, input_w)))           # 添加解码器
    
    result = {
        'objects': []
    }

    with torch.no_grad():
        res = exec_net.infer(inputs={input_blob: images})
        output1 = torch.from_numpy(res[out_blob])
        output2 = torch.from_numpy(res[next(iterator)])
        output3 = torch.from_numpy(res[next(iterator)])
        outputs = [output1, output2, output3]           # 5D, torch

        output_list = []
        for i in range(3):
            output_list.append(yolo_decodes[i](outputs[i]))            # 先解码，获得像素坐标值，再堆叠

        #---------------------------------------------------------#
        #   将预测框进行堆叠，然后进行非极大抑制
        #---------------------------------------------------------#
        output = torch.cat(output_list, 1)
        batch_detections = non_max_suppression(output, num_classes, conf_thres=confidence, nms_thres=iou)
        batch_detections = batch_detections[0]
        if batch_detections is None:
            return json.dumps(result, indent=4)
        #---------------------------------------------------------#
        #   对预测框进行得分筛选
        #---------------------------------------------------------#
        top_index   = batch_detections[:, 4] * batch_detections[:, 5] > confidence
        if top_index is None:
            return json.dumps(result, indent=4)
        top_conf    = batch_detections[top_index, 4] * batch_detections[top_index, 5]
        top_label   = np.array(batch_detections[top_index, -1], np.int32)           # 数字标签，在 non_max_suppression中
        top_bboxes  = np.array(batch_detections[top_index, :4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

        top_xmin = top_xmin / model_image_size[1] * image_shape[1]
        top_ymin = top_ymin / model_image_size[0] * image_shape[0]      # 注意图片尺寸定义差异
        top_xmax = top_xmax / model_image_size[1] * image_shape[1]
        top_ymax = top_ymax / model_image_size[0] * image_shape[0]
        boxes = np.concatenate([top_xmin, top_ymin, top_xmax, top_ymax], axis=-1)
    
    for i, c in enumerate(top_label):
        xmin, ymin, xmax, ymax = boxes[i]
        name = class_names[c]
        confidence = top_conf[i]
        result['objects'].append({
            'xmin':int(xmin),
            'ymin':int(ymin),
            'xmax':int(xmax),
            'ymax':int(ymax),
            'name':str(name),
            'confidence':float(confidence)
        })
    return json.dumps(result, indent=4)
# end


if __name__ == '__main__':
    # Test API
    img = cv2.imread('/project/ev_sdk/data/bluesky.jpg')
    predictor = init()
    result = process_image(predictor, img)
    print(result)
