# darknet-server
A yolov7 (pytorch-darknet) object detection socket server

## Requirements
- `docker`
- `docker-compose`
- `nvidia-container-toolkit`, if you want to take advantage of the gpu
- `python3`, if you want to use the sample client -- else you can write your own client in whatever language

## Running
First, install & run the server:
 - ```docker-compose up```

Then connect your client (sample client provided):
- ```python code/client.py```

## Server-Client Protocol
1. Server receives 8 bytes, representing file size N in big-endian
2. Server receives N additional bytes from the client, containing raw bytes of image (image file contents in 'rb' mode)
3. Server performs detection and sends back the list of detections with bounding boxes, formatted as [timestamp, (label, confidence, (x, y, width, height)),,...]
4. Repeat 1-3 as many times as desired over one connection until a size of 0 bytes is sent, at which point the connection will terminate.

The server can conect to multiple clients at once and will process images in the order received.
