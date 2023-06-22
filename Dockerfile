FROM nvcr.io/nvidia/pytorch:21.08-py3
RUN apt update
RUN apt install -y zip htop screen libgl1-mesa-glx
RUN pip install seaborn thop
WORKDIR /models
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt 
WORKDIR /code
CMD ["python", "server.py"]
#WORKDIR yolov7
#CMD ["python", "detect.py", "--source", "/images/test.jpg"]
