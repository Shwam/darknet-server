from datetime import datetime
import os
import time
import sys

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

sys.path.insert(0, "./yolov7")
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized

def recv_image():
    # Receive an image over the network
    received = datetime.now().strftime("%Y%m%d-%H%M%S.jpg")
    img0 = cv2.imread("/images/test.jpg")  # BGR
    return img0, received

def load_image(img_size, stride, device, img0):
    # Padded resize
    img = letterbox(img0, img_size, stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img, img0 # RGB, BGR



def server(weights="/models/yolov7.pt", img_size=640,  conf_thresh=0.25, iou_thresh=0.45, classes=None):
    # Check if weights exist
    if not (os.path.exists(weights)):
        default_weights = "/models/yolov7.pt"
        if not (os.path.exists(default_weights)):
            print(f"Downloading {default_weights}")
            os.system("cd /models/ && wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt")
        weights = default_weights
        if weights != default_weights:
            print(f"Could not find weights file {weights}, using default weights {default_weights}")

    # Initialize
    set_logging()
    device = select_device("")
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check img_size
    
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once

    while True:
        image, received = recv_image()
        time.sleep(0.5) # We don't need 20 images per second...
        start = time.time()
        
        # Load image from memory
        img, im0s = load_image(img_size, stride, device, image)
    
        detections = detect(model, img, im0s, conf_thresh, iou_thresh, classes, names)
        
        #output_queue.put(detections)
        print(f"Processed image in {time.time() - start} seconds")

        print(detections)
        # save image and xml
        cv2.imwrite("/output/" + received, (im0s).astype('uint8')) 
        save_xml(detections, received)
    
    

def detect(model, img, im0s, conf_thresh, iou_thresh, classes, names):
    # Inference
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=classes, agnostic=False)

    detections = [] # (class, x, y, w, h, confidence)

    # Process detections
    for i, det in enumerate(pred):  # detections per image

        gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            # normalize xywh 
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
                detections.append((names[int(cls)], float(conf), xywh))

    return detections

def save_xml(boxes, filename):
    width = 2560
    height = 1440
    path = f"/output/{filename}"
    object_template = lambda label, rect: f"""<object>
        <name>{label}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{int((rect[0]-rect[2]/2)*width)}</xmin>
            <ymin>{int((rect[1]-rect[3]/2)*height)}</ymin>
            <xmax>{int((rect[0]+rect[2]/2)*width)}</xmax>
            <ymax>{int((rect[1]+rect[3]/2)*height)}</ymax>
        </bndbox>
    </object>"""
    objects = ""
    for box in boxes:
        label, confidence, rect = box    
        objects += object_template(label, rect) + "\n"
                
    output = f"""<annotation>
        <folder>images</folder>
        <filename>{filename}</filename>
        <path>{path}</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>{width}</width>
            <height>{height}</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
        {objects}
    </annotation>"""

    with open(path.replace(".jpg", ".xml"), "w") as f:
        f.write(output)

if __name__ == '__main__':
    with torch.no_grad():
        server()
