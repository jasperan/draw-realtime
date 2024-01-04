# Copyright (c) 2022 Oracle and/or its affiliates.

'''
@author jasperan
This file uses web sockets to communicate Live Client API data from one endpoint to the other. You can virtually put your consumer 
wherever you want, and let it process the incoming data from the producer.
''' 

import requests
import datetime
import argparse
import json 
import websockets
import asyncio
from rich import print
import os
import sys
import time
import threading
from multiprocessing import Process, Queue, get_context
from typing import List, Literal, Dict, Optional
import torch
import PIL.Image
from streamdiffusion.image_utils import pil2tensor
import mss
import fire
import tkinter as tk
from viewer import receive_images
from utils.wrapper import StreamDiffusionWrapper
import base64
from io import BytesIO


def encode_msg(msg: Dict) -> str:
    return json.dumps(msg, ensure_ascii=False)



inputs = []
top = 0
left = 0

global sct
sct = mss.mss()



def screen(
    height: int = 1024,
    width: int = 1024,
    monitor: Dict[str, int] = {"top": 300, "left": 200, "width": 1024, "height": 1024},
):
    global inputs
    global sct
    with sct:
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

        print(im_b64)

        return im_b64
            
            
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



cli_parser = argparse.ArgumentParser()
cli_parser.add_argument('-i', '--ip', type=str, help='IP address to make requests to',
                        default='127.0.0.1',
                        required=False)
args = cli_parser.parse_args()

# start screen capture

height: int = 1024
width: int = 1024
monitor: Dict[str, int] = {"top": 100, "left": 100, "width": 1024, "height": 1024}


'''
event = threading.Event()
input_screen = threading.Thread(target=screen, args=(event, height, width, monitor))
input_screen.start()
time.sleep(5)
'''


async def handler(websocket):
    while True:
        message = await websocket.recv()
        print(message)

        if message == 'get_img':
            result = screen(height, width, monitor)
        else:
            result = {}

        await websocket.send(result)





async def main():
    async with websockets.serve(handler, "", 8001):
        await asyncio.Future()  # run forever
    #event.set() # stop capture thread
    #input_screen.join()


if __name__ == "__main__":
    asyncio.run(main())

'''
def main(
    model_id_or_path: str = "KBlueLeaf/kohaku-v2.1",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "1girl with brown dog hair, thick glasses, smiling",
    negative_prompt: str = "low quality, bad quality, blurry, low resolution",
    frame_buffer_size: int = 1,
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
    seed: int = 2,
    cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    guidance_scale: float = 1.4,
    delta: float = 0.5,
    do_add_noise: bool = False,
    enable_similar_image_filter: bool = True,
    similar_image_filter_threshold: float = 0.99,
    similar_image_filter_max_skip_frame: float = 10,
) -> None:
    """
    Main function to start the image generation and viewer processes.
    """
    ctx = get_context('spawn')
    queue = ctx.Queue()
    fps_queue = ctx.Queue()
    close_queue = Queue()


    process = ctx.Process(target=receive_images, args=(queue, fps_queue))
    process.start()

    # terminate
    process.join()
    print("process2 terminated.")
    close_queue.put(True)
'''