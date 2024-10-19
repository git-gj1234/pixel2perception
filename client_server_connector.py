import socket
import time
import io
from PIL import Image

def send_image_and_text(image, text_input):
    host = '10.242.57.219'  
    send_port = 12345
    receive_port = 12346

    s = socket.socket()
    s.connect((host, send_port))

    text_data = text_input.encode()
    text_size = len(text_data)
    s.sendall(str(text_size).encode().ljust(16))  
    s.sendall(text_data)  

    with io.BytesIO() as image_bytes:
        image.save(image_bytes, format='PNG')
        image_data = image_bytes.getvalue()

    image_size = len(image_data)
    s.sendall(str(image_size).encode().ljust(16))  
    s.sendall(image_data) 
    s.close()

    r = socket.socket()
    r.bind(('0.0.0.0', receive_port))
    r.listen(1)

    conn, addr = r.accept()

    output_data = b""
    while True:
        packet = conn.recv(4096)
        if not packet:
            break
        output_data += packet
    
    conn.close()
    r.close()

    return output_data.decode()