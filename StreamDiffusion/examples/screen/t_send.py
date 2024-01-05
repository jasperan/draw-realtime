import websockets
from rich import print
import os
import sys
import time
import threading
from multiprocessing import Process, Queue, get_context
from typing import List, Literal, Dict, Optional
import torch
import PIL.Image
import mss
import tkinter as tk
import base64
from io import BytesIO
import asyncio
# this code sends base64 images periodically after every screenshot to t_ss.py
b64_img = "Hello"

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
            print(type(img_data), len(img_data))
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
            b64_img = im_b64
            print("[x] Sent Base64 image")

            #print(im_b64)

            return im_b64
            
        connection.close()  
        print('exit : screen')


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


async def main():
    print(1)
    width: int = 1024
    height: int = 1024
    monitor = dummy_screen(width, height)
    print(1)

    event = threading.Event()
    input_screen = threading.Thread(target=screen, args=(event, height, width, monitor))
    input_screen.start()
#time.sleep(5)
    print(1)

    while True:
        print(2)
        global b64_img
        async with websockets.connect('ws://127.0.0.1:8765') as websocket:
            await websocket.send(b64_img)
            response = await websocket.recv()
            print('Received {}'.format(response))
            
        time.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())