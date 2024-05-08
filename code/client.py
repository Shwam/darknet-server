import ast
from socket import *
from struct import pack
import multiprocessing, threading
from queue import Empty
import time

RECV_SIZE=4096

class DarknetClient:

    def __init__(self, server_ip, server_port, image_queue, result_queue):
        self.socket = None
        self.server_ip = server_ip
        self.server_port = server_port
        self.image_queue = image_queue
        self.result_queue = result_queue

    def connect(self):
        retries = 0
        while True:
            try:
                self.socket = socket(AF_INET, SOCK_STREAM)
                self.socket.connect((self.server_ip, self.server_port))
                print(f"Connected to image server at {self.server_ip}:{self.server_port}")
                return True
            except Exception as err:
                if retries == 0:
                    print(f"Could not connect to {self.server_ip}:{self.server_port} - {err}, retrying")
                time.sleep(5)
                retries += 1
    
    def run(self):
        self.connect()
        data = b""
        while True:
            image = self.image_queue.get(block=True)
            if image is None: # quit command
                self.close()
                return 0
            # send it to the server
            try:
                self.send_image(image)
            except Exception as err:
                print(f"Error communicating with image server {err}. Retrying connection")
                self.connect()
                self.image_queue.put(image)
                continue
            data = b""
            while not data:
                try:
                    data = self.socket.recv(RECV_SIZE)
                    if not data:
                        print(f"Error communicating with image server (Empty data). Retrying connection")
                        self.connect()
                        break
                except Exception as err:
                    print(f"Error communicating with image server {err}. Retrying connection")
                    self.connect()
                    break
                if data:
                    self.result_queue.put(ast.literal_eval(data.decode("utf8")))
                else:
                    print("No data returned from server")
                    # Try it again
                    self.image_queue.put(image)
    
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
