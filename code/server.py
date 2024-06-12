from datetime import datetime
import os
import time
import sys
import socket
import threading, multiprocessing
from queue import Empty
from struct import unpack
from io import BytesIO

from PIL import Image
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

RECV_SIZE=4096

sys.path.insert(0, "./yolov7")
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized

class DarknetServer(socket.socket):
    clients = dict()

    def __init__(self, image_queue, address="0.0.0.0", port=7061):
        self.image_queue = image_queue
        socket.socket.__init__(self)
        self.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.bind((address, port))
        self.address = address
        self.port = port
        self.listen(5)

    def run(self):
        print("Started server")
        try:
            self.accept_clients()
        except Exception as err:
            print(err)
        finally:
            print("Closing server")
            for client in self.clients:
                client.close()
            self.close()

    def accept_clients(self):
        while True:
            conn, address = self.accept()
            self.clients[conn] = (address, None)
            # Client Connected
            self.onopen(conn, address)
            # Begin communicating with the client
            threading.Thread(target=self.receive, args=(conn,)).start()

    def receive(self, client):
        # Continuously communicate with client
        try:
            while True:
                byte_s = client.recv(8) # First byte = file size
                (length,) = unpack('>Q', byte_s)
                if length == 0: # Termination command
                    break
                received = datetime.now().strftime("%y%m%d%H%M%S%f")
                data = b""
                while len(data) < length:
                        to_read = length - len(data)
                        data += client.recv(RECV_SIZE if to_read > RECV_SIZE else to_read)
                
                # add to image queue
                self.image_queue.put((client, data, received))
                
        except Exception as err:
            print("Error while communicating with client", err)
        # close thread
        del self.clients[client]
        self.onclose(client)
        client.close()
        sys.exit()

    def broadcast(self, message):
        for client in self.clients:
            client.send(message)

    def onopen(self, client, addr):
        print(f"Client connected from {addr}")
 
    def onclose(self, client):
        print("Client Disconnected")

def recv_image(input_queue):
    # Receive an image over the network
    command = input_queue.get(block=True, timeout=None)
    return command # client, image, received

def load_image(img_size, stride, device, memory_image):
    # Load from bytes
    img0 = np.asarray(Image.open(BytesIO(memory_image)))
    
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



def detector_thread(input_queue, output_queue, weights="/models/yolov7.pt", img_size=640,  conf_thresh=0.85, iou_thresh=0.85, classes=None):
    
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
        try:
            client, image, received = recv_image(input_queue)
            # Load image from memory
            img, im0s = load_image(img_size, stride, device, image)
            start = time.time()
            detections = detect(model, img, im0s, conf_thresh, iou_thresh, classes, names)
            #print(f"Processed image in {time.time() - start} seconds")
            #print(f"Last activity: {datetime.now().strftime('%y%m%d%H%M%S%f')}")
            client.send(str((received, detections)).encode("utf8"))
        except Exception as err:
            print(f"Failed to process image from client {client.getpeername()}: {err}") 
    

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

if __name__ == '__main__':
    with torch.no_grad():
        input_queue = multiprocessing.Queue()
        output_queue = multiprocessing.Queue()

        # start detection worker thread
        threading.Thread(target=detector_thread, args=[input_queue, output_queue]).start()

        # start server
        server = DarknetServer(input_queue)
        server.run()
        
