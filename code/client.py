import ast
from socket import *
from struct import pack
import multiprocessing, threading
from queue import Empty

RECV_SIZE=4096

class DarknetClient:

    def __init__(self, server_ip, server_port, image_queue, result_queue):
        self.socket = None
        self.connect(server_ip, server_port)
        self.image_queue = image_queue
        self.result_queue = result_queue

    def connect(self, server_ip, server_port):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.connect((server_ip, server_port))
    
    def run(self):
        data = b""
        while True:
            try:
                image = self.image_queue.get(block=False)
                if image == None: # quit command
                    self.close()
                    break
                if image:
                    # send it to the server
                    self.send_image(image)
                data = self.socket.recv(RECV_SIZE)
                if data:
                    self.result_queue.put(ast.literal_eval(data.decode("utf8")))
            except Empty:
                pass
        
    def close(self):
        # send the quit command
        self.socket.sendall(pack('>q', 0))
        # shut down
        self.socket.shutdown(SHUT_WR)
        self.socket.close()
        self.socket = None

    def send_image(self, image_data):
        length = pack('>Q', len(image_data))
        self.socket.sendall(length)
        self.socket.sendall(image_data)
if __name__ == '__main__':
    """ Sample client which performs detection on a single image and then disconnects """
    # Start the client thread
    image_queue = multiprocessing.Queue() # put images to be processed
    result_queue = multiprocessing.Queue() # receive detections
    client = DarknetClient('localhost', 7061, image_queue, result_queue)
    threading.Thread(target=DarknetClient.run, args=(client,)).start()

    # Queue an image to be processed
    with open('test.jpg', 'rb') as f:
        image_data = f.read()    
        image_queue.put(image_data)

    # Get the resulting detection
    result = result_queue.get(block=True, timeout=None)
    print("Got detection", result)

    # Tell the client to shut down
    image_queue.put(None)
