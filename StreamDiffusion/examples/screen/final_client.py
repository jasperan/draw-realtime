import os
import sys
import time
import threading
from multiprocessing import Process, Queue, get_context
from typing import List, Literal, Dict, Optional
import torch
from streamdiffusion.image_utils import pil2tensor
import fire
import tkinter as tk
from streamdiffusion.image_utils import postprocess_image
from PIL import Image
from io import BytesIO
import base64
import websockets
import asyncio
import requests
import requests
from requests.adapters import HTTPAdapter
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.viewer import receive_images
from utils.wrapper import StreamDiffusionWrapper

inputs = []

# this variable will be updated every time a new message comes from the queue.
global_img = []


def screen(
    event: threading.Event,
    height: int = 2560,
    width: int = 1440,
    monitor: Dict[str, int] = {"top": 0, "left": 0, "width": 2560, "height": 1440},
):
    global inputs
    while True:
        if event.is_set():
            print("terminate read thread")
            break
        # this works img = Image.open("img/forgh3.jpg")
        #global_img = open('a.base64', 'r').read()
        global global_img
        s = requests.Session()
        retries = Retry(total=5,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504])
        
        s.mount('http://', HTTPAdapter(max_retries=retries))

        response = s.get('http://127.0.0.1:8000')
        #print(response.text)
        data = response.json()
        assert type(data) == type(list())

        if len(data) > 1:
            print('Data Obtained Size {}'.format(len(data)))

        for x in data:
            im_bytes = base64.b64decode(x)
            #im_bytes = base64.b64decode(global_img)   # im_bytes is a binary image
            im_file = BytesIO(im_bytes)  # convert image to file-like object
            img = Image.open(im_file)   # img is now PIL Image object
            img.resize((height, width))

            # img object is our image

            inputs.append(pil2tensor(img))
        # after all iterations are done, clear our list of base64 encodings after they've been added 
        # to the q.
        global_img.clear()

        #time.sleep(2)
        #print(inputs)
    print('exit : screen')
def dummy_screen(
        width: int,
        height: int,
):
    root = tk.Tk()
    root.title("draw-realtime by jasperan")
    root.geometry(f"{width}x{height}")
    root.resizable(True, True)
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

def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
    close_queue: Queue,
    model_id_or_path: str,
    lora_dict: Optional[Dict[str, float]],
    prompt: str,
    negative_prompt: str,
    frame_buffer_size: int,
    width: int,
    height: int,
    acceleration: Literal["none", "xformers", "tensorrt"],
    use_denoising_batch: bool,
    seed: int,
    cfg_type: Literal["none", "full", "self", "initialize"],
    guidance_scale: float,
    delta: float,
    do_add_noise: bool,
    enable_similar_image_filter: bool,
    similar_image_filter_threshold: float,
    similar_image_filter_max_skip_frame: float,
    monitor: Dict[str, int],
) -> None:
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    queue : Queue
        The queue to put the generated images in.
    fps_queue : Queue
        The queue to put the calculated fps.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    negative_prompt : str, optional
        The negative prompt to use.
    frame_buffer_size : int, optional
        The frame buffer size for denoising batch, by default 1.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"], optional
        The acceleration method, by default "tensorrt".
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    cfg_type : Literal["none", "full", "self", "initialize"],
    optional
        The cfg_type for img2img mode, by default "self".
        You cannot use anything other than "none" for txt2img mode.
    guidance_scale : float, optional
        The CFG scale, by default 1.2.
    delta : float, optional
        The delta multiplier of virtual residual noise,
        by default 1.0.
    do_add_noise : bool, optional
        Whether to add noise for following denoising steps or not,
        by default True.
    enable_similar_image_filter : bool, optional
        Whether to enable similar image filter or not,
        by default False.
    similar_image_filter_threshold : float, optional
        The threshold for similar image filter, by default 0.98.
    similar_image_filter_max_skip_frame : int, optional
        The max skip frame for similar image filter, by default 10.
    """
    
    global inputs
    taesd_model = "madebyollin/taesd"
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[32, 45],
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
        use_tiny_vae=True, # TAESD
        warmup=10,
        acceleration=acceleration,
        do_add_noise=do_add_noise,
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=similar_image_filter_threshold,
        similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame,
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type=cfg_type,
        seed=seed,
    )

    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        delta=delta,
    )
    event = threading.Event()
    input_screen = threading.Thread(target=screen, args=(event, height, width, monitor))
    input_screen.start()
    #time.sleep(5)

    while True:
        try:
            if not close_queue.empty(): # closing check
                break
            if len(inputs) < frame_buffer_size:
                time.sleep(0.005)
                continue
            start_time = time.time()
            sampled_inputs = []
            for i in range(frame_buffer_size):
                index = (len(inputs) // frame_buffer_size) * i
                sampled_inputs.append(inputs[len(inputs) - index - 1])
            input_batch = torch.cat(sampled_inputs)
            inputs.clear()
            output_images = stream.stream(
                input_batch.to(device=stream.device, dtype=stream.dtype)
            ).cpu() # this step processes the input batch
            if frame_buffer_size == 1:
                output_images = [output_images] # we get the results already
            for output_image in output_images:
                #print(type(output_image))
                #print(output_image)
                #print('Queue size: {}'.format(queue.qsize()))
                #res = postprocess_image(output_image, output_type="pil")[0]
                #print(res, type(res))
                #res.save('exxx.png')
                #time.sleep(3)
                queue.put(output_image, block=False)

            fps = 1 / (time.time() - start_time)
            fps_queue.put(fps)
        except KeyboardInterrupt:
            break

    print("closing image_generation_process...")
    event.set() # stop capture thread
    input_screen.join()
    print(f"fps: {fps}")

def main(
    #model_id_or_path: str = "KBlueLeaf/kohaku-v2.1",
    model_id_or_path: str ="stabilityai/sd-turbo",
    lora_dict: Optional[Dict[str, float]] = None,
    #prompt: str = "1girl with brown dog hair, thick glasses, smiling",
    #prompt: str = "1girl, anime, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details",
    prompt: str = "landscape, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details",
    negative_prompt: str = "low quality, bad quality, blurry, low resolution, pixelated, pixel art, low fidelity",
    frame_buffer_size: int = 1,
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
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

    def callback(ch, method, properties, body):
        global global_img
        print(f" [x] Received {body}")
        global_img = body

    start_server = websockets.serve(callback, "localhost", 8001)
    asyncio.get_event_loop().run_until_complete(start_server)


    monitor = dummy_screen(width, height)
    ctx = get_context('spawn')
    queue = ctx.Queue()
    fps_queue = ctx.Queue()
    close_queue = Queue()
    process1 = ctx.Process(
        target=image_generation_process,
        args=(
            queue,
            fps_queue,
            close_queue,
            model_id_or_path,
            lora_dict,
            prompt,
            negative_prompt,
            frame_buffer_size,
            width,
            height,
            acceleration,
            use_denoising_batch,
            seed,
            cfg_type,
            guidance_scale,
            delta,
            do_add_noise,
            enable_similar_image_filter,
            similar_image_filter_threshold,
            similar_image_filter_max_skip_frame,
            monitor
            ),
    )
    process1.start()

    process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
    process2.start()





    print(' [*] Waiting for messages. To exit press CTRL+C')
    

    # terminate
    process2.join()
    print("process2 terminated.")
    close_queue.put(True)
    print("process1 terminating...")
    process1.join(5) # with timeout
    if process1.is_alive():
        print("process1 still alive. force killing...")
        process1.terminate() # force kill...
    process1.join()
    print("process1 terminated.")


if __name__ == "__main__":
    fire.Fire(main)