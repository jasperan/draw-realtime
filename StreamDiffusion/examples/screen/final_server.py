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
import argparse

# this code sends base64 images periodically after every screenshot to t_ss.py
b64_img = []


# import monitor settings in settings.py
from settings import settings_2k, settings_4k, settings_1080p, settings_720p, settings_480p


# ask for command line parameters
cli_parser = argparse.ArgumentParser()
cli_parser.add_argument('-i', '--ip', type=str, help='IP address to make requests to',
                        default='127.0.0.1',
                        required=False)

cli_parser.add_argument('-m', '--mode',
                        type=str,
                        help='Execution Mode',
                        default='2k',
                        choices=['480p, 720p, 1080p, 2k, 4k'],
                        required=False)

args = cli_parser.parse_args()


app = FastAPI()


def screen(
    event: threading.Event,
    height: int,
    width: int,
    monitor: Dict[str, int],
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
        top: int,
        left: int,
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


def get_settings(execution_mode):
    if execution_mode == '480p':
        return settings_480p
    elif execution_mode == '720p':
        return settings_720p
    elif execution_mode == '1080p':
        return settings_1080p
    elif execution_mode == '2k':
        return settings_2k
    elif execution_mode == '4k':
        return settings_4k
    else: return {}


def main():

    width: int = 1024
    height: int = 1024

    #get_settings(args.mode)
    # Now, we don't need the dummy screen anymore:
    #monitor = dummy_screen(top, left, width, height)
    monitor = get_settings(args.mode)

    #while True:
    '''
    global b64_img
    async with websockets.connect('ws://127.0.0.1:8765') as websocket:
        await websocket.send(b64_img)
        response = await websocket.recv()
        print('Received {}'.format(response))
    '''

    
    event = threading.Event()
    input_screen = threading.Thread(target=screen, args=(event,
        monitor['height'],
        monitor['width'],
        monitor))
    input_screen.start()
    
        
    input_screen.join()

@app.get("/")
async def root():
    global b64_img
    
    returner = list(b64_img) # list type

    
    #print('Size before clear: {}'.format(len(b64_img)))
    b64_img.clear()
    #print('Size after clear: {} vs. |{}|'.format(len(b64_img), len(returner)))
    return returner

def run_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
if __name__ == "__main__":
    uvicorn_thread = threading.Thread(target=run_uvicorn)
    uvicorn_thread.start()
    main()




