# Copyright (c) 2022 Oracle and/or its affiliates.

'''
@author jasperan
This file uses web sockets to communicate Live Client API data from one endpoint to the other. You can virtually put your consumer 
wherever you want, and let it process the incoming data from the producer.
''' 
import asyncio
from websockets import connect
import sys, os
import argparse
#from rich import print
import base64
import cv2
import numpy as np
from multiprocessing import Process, Queue, get_context
from typing import List, Literal, Dict, Optional
import time
from wrapper import StreamDiffusionWrapper
from streamdiffusion.image_utils import pil2tensor #  https://github.com/cumulo-autumn/StreamDiffusion/blob/71932007026d1e6b85d53186b2e21b92456ab0cc/src/streamdiffusion/image_utils.py#L87
import torch
from io import BytesIO
from PIL import Image

from viewer import receive_images
from streamdiffusion.image_utils import postprocess_image
import torchvision.transforms as T
from torchvision.utils import save_image

import matplotlib.pyplot as plt

cli_parser = argparse.ArgumentParser()
cli_parser.add_argument('-i', '--ip', type=str, help='IP address to make requests to', default='127.0.0.1', required=False)
args = cli_parser.parse_args()


def display_base_64(img_b64):
    decoded_data = base64.b64decode(img_b64)
    np_data = np.fromstring(decoded_data,np.uint8)
    img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    #cv2.imshow("test", img)
    #cv2.waitKey(0)

def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
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
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[32, 45],
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
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

    try:
        if len(inputs) < frame_buffer_size:
            time.sleep(0.005)
        start_time = time.time()
        sampled_inputs = []
        '''
        for i in range(frame_buffer_size):
            index = (len(inputs) // frame_buffer_size) * i
            sampled_inputs.append(inputs[len(inputs) - index - 1])
        
        
        #test
        '''
        sampled_inputs = [input]
        input_batch = torch.cat(sampled_inputs)
        #assert sampled_inputs == inputs
        
        #assert len(sampled_inputs) == len(inputs)
        #assert len(input_batch) == len(inputs)
        #print(len(sampled_inputs), len(inputs), len(input_batch))
        #inputs.clear()
        #print('4')
        output_images = stream.stream(
            input_batch.to(device=stream.device, dtype=stream.dtype)
        ).cpu() # this step processes the input batch
        #print('5')
        if frame_buffer_size == 1:
            output_images = [output_images] # we get the results already
        for output_image in output_images:
            print(type(output_image))
            #print('Found output image. Displaying... {}'.format(len(output_image)))
            #cv2.imshow("test", output_image)
            #cv2.waitKey(0)
            #queue.put(output_image, block=False)
            #print('Queue size: {}'.format(queue.qsize()))
            print(output_image)
            #save_image(output_image, 'out.png')
            res = postprocess_image(output_image, output_type="pil")[0]
            print(res, type(res))
            #print(np.asarray(res))
            res.save('exxx.png')
            time.sleep(1)
            #postprocess_image(queue.get(block=False), output_type="pil")[0]
            #display_base_64(output_images)
            '''
            transform = T.ToPILImage()
            
            img = transform(output_image[0])
            '''
            #print(type(img))
            #img.save('img/test.jpg')
            #plt.imshow(img)
            #plt.show() # image will not be displayed without this

        fps = 1 / (time.time() - start_time)
    except KeyboardInterrupt:
        print('KeyboardInt')

    print("Closing image_generation_process...")
    print(f"fps: {fps}")


inputs = []
input = Image.Image

def process_image(img_b64, width, height):
    im_bytes = base64.b64decode(img_b64)   # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
    img = Image.open(im_file)   # img is now PIL Image object
    img.resize((height, width))
    #plt.imshow(img)
    #plt.show()

    #decoded_data = base64.b64decode(img_b64)
    #np_data = np.fromstring(decoded_data,np.uint8)
    #img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)

    # TODO https://github.com/cumulo-autumn/StreamDiffusion/blob/71932007026d1e6b85d53186b2e21b92456ab0cc/src/streamdiffusion/image_utils.py#L87
    #img = PIL.Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
    #img = Image.frombytes("RGB", (width, height), im_file, "raw", "BGRX")
    #img = Image.fromarray(im_file)
    #img.resize((height, width))
    #global inputs
    #inputs.append(pil2tensor(img))
    global input
    pilimage = pil2tensor(img)
    input = pilimage
    #global inputs
    #inputs.append(input)


    print('GLOBAL INPUTS SIZE {}'.format(len(inputs)))
    #print(pil2tensor(img)
    

async def get_info(uri,
    model_id_or_path: str = "KBlueLeaf/kohaku-v2.1",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "1girl with brown dog hair, thick glasses, smiling",
    negative_prompt: str = "low quality, bad quality, blurry, low resolution",
    frame_buffer_size: int = 1,
    width: int = 1024,
    height: int = 1024,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
    seed: int = 2,
    cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    guidance_scale: float = 1.4,
    delta: float = 0.5,
    do_add_noise: bool = False,
    enable_similar_image_filter: bool = True,
    similar_image_filter_threshold: float = 0.99,
    similar_image_filter_max_skip_frame: float = 10
) -> None:
    async with connect(uri) as websocket:
        ctx = get_context('spawn')
        fps_queue = ctx.Queue()
        queue = ctx.Queue()

        #process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
        #process2.start()


        print("process2 terminated.")
        while True:
            await websocket.send("get_img")
            img_b64 = await websocket.recv()
            print('Received Image')
            #print(img_b64)
            
            #display_base_64(img_b64)    
            process_image(img_b64, width, height)

            #display_base_64(img_b64)

            print('Image Generation Starting...')
            image_generation_process(
                queue,
                fps_queue,
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
            )
            print('Image Generation Finished')
        # terminate
        #process2.join()
            
            


def main():
    asyncio.run(get_info('ws://{}:8001'.format(args.ip)))



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
