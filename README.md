# darknet-server
A yolov7 (pytorch-darknet) object detection socket server

## Requirements
- `nvidia-docker` (if you want to take advantage of the gpu)
- `docker-compose`
- `python3` (if you want to use the sample client, else you can write your own client in whatever language)
## Installing and Running
First, run the server:
 - ```docker-compose up```

Then connect your client (sample client provided):
- ```python code/client.py```
