from rich import print
import threading
from multiprocessing import Process, Queue, get_context
from typing import List, Literal, Dict, Optional
import PIL.Image
import mss
import tkinter as tk
import base64
from io import BytesIO
from fastapi import FastAPI
import uvicorn
import time

# this code sends base64 images periodically after every screenshot to t_ss.py
b64_img = []



app = FastAPI()


def screen(
    event: threading.Event,
    height: int = 1024,
    width: int = 1024,
    monitor: Dict[str, int] = {"top": 300, "left": 200, "width": 1024, "height": 1024},
):
    global inputs
    global sct

    while True:
        with mss.mss() as sct:
            img = sct.grab(monitor)
            # img object is our image
            img_data = img.bgra
            img = PIL.Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
            #print(type(img_data), len(img_data))
            #img.resize((height, width))


            # img object is our image
            #print('Image Size: {}'.format(img.size))
            #img.save('local_image.png') save the image

            im_file = BytesIO()
            img.save(im_file, format="JPEG")
            im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
            im_b64 = base64.b64encode(im_bytes)

            # SEND DATA
            global b64_img
            b64_img.append(im_b64)
            #b64_img.append("Hello")
            #print("[x] Sent Base64 image")

            #print(im_b64)
            # write b64_img to a local file called data.base64
            #mutex = Lock()

            #mutex.acquire()
            #with open('data.base64', 'wb') as file:
            #    file.write(im_b64)
            #mutex.release()

            print('Buffer Length: {} | Image 1 Length: {}'.format(len(b64_img), len(b64_img[0])))
            time.sleep(.5)
            #return b64_img
            


def dummy_screen(
        width: int,
        height: int,
):
    root = tk.Tk()
    root.title("Press Enter to start")
    root.geometry(f"{width}x{height}")
    root.resizable(False, False)
    root.attributes("-alpha", 0.8)
    root.configure(bg="black")
    def destroy(event):
        root.destroy()
    root.bind("<Return>", destroy)
    def update_geometry(event):
        global top, left
        top = root.winfo_y()
        left = root.winfo_x()
    root.bind("<Configure>", update_geometry)
    root.mainloop()
    return {"top": top, "left": left, "width": width, "height": height}


def main():
    width: int = 1024
    height: int = 1024
    monitor = dummy_screen(width, height)

    event = threading.Event()
    input_screen = threading.Thread(target=screen, args=(event, height, width, monitor))
    input_screen.start()

    #while True:
    '''
    global b64_img
    async with websockets.connect('ws://127.0.0.1:8765') as websocket:
        await websocket.send(b64_img)
        response = await websocket.recv()
        print('Received {}'.format(response))
    '''
    event = threading.Event()
    input_screen = threading.Thread(target=screen, args=(event, height, width, monitor))
    input_screen.start()
        
    input_screen.join()

@app.get("/")
async def root():
    global b64_img
    returner_val = b64_img # list type
    
    returner = list(b64_img)    


    #print(returner)
    print('Size before clear: {}'.format(len(b64_img)))
    b64_img.clear()
    print('Size after clear: {} vs. |{}|'.format(len(b64_img), len(returner)))
    return returner

def run_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
if __name__ == "__main__":
    uvicorn_thread = threading.Thread(target=run_uvicorn)
    uvicorn_thread.start()
    main()




