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
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.viewer import receive_images
from utils.wrapper import StreamDiffusionWrapper

inputs = []
final_images = []

# will store the image size of the video being processed.
width: int = 512
height: int = 512



def screen():
    global inputs
    global width, height
    vidcap = cv2.VideoCapture('video/video.mp4')
    success,image = vidcap.read()
    count = 0
    if width == 512:
        width = image.shape[1]
        height = image.shape[0]
    print('Video Properties: {}x{}'.format(image.shape[1], image.shape[0]))    
    while success:
        if count < 10:
            #print(type(image))
            #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
            success, image = vidcap.read()
            try:
            #print('Read a new frame: ', success)
            # img object is our image
                pil_img = Image.fromarray(image)
            except (AttributeError):
                count+= 1
                continue
            #print(type(pil_img))
            inputs.append(pil2tensor(pil_img))
            count += 1
            print('Frame {}. Inputs length: {}'.format(count, len(inputs)))
        else: break
    print('exit : screen')


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
    similar_image_filter_max_skip_frame: float
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
    input_screen = threading.Thread(target=screen, args=(event, height, width))
    input_screen.start()

    while True:
        try:
            if not close_queue.empty(): # closing check
                break
            global inputs
            if len(inputs) < frame_buffer_size:
                time.sleep(0.005)
            start_time = time.time()
            sampled_inputs = []
            #for i in range(frame_buffer_size):
            #    index = (len(inputs) // frame_buffer_size) * i
            #    sampled_inputs.append(inputs[len(inputs) - index - 1])
                #print(index, len(inputs) - index - 1, (len(inputs) // frame_buffer_size)*i)
            sampled_inputs = inputs.copy()
            #print(type(sampled_inputs[0]))
            input_batch = torch.cat(sampled_inputs)
            print('Inputs before clear: {} {} {}'.format(len(inputs), len(input_batch), len(sampled_inputs)))
            inputs.clear()
            print('Inputs after clear: {}'.format(len(inputs)))
            output_images = stream.stream(
                input_batch.to(device=stream.device, dtype=stream.dtype)
            ).cpu() # this step processes the input batch
            if frame_buffer_size == 1:
                output_images = [output_images] # we get the results already
            for output_image in output_images:
                queue.put(output_image, block=False)
                #print('Queue put image {}'.format(output_image))
            print('Output images size: {}'.format(len(output_images)))
            fps = 1 / (time.time() - start_time)
            fps_queue.put(fps)
        except KeyboardInterrupt:
            pass

    print("closing image_generation_process...")
    print(f"fps: {fps}")


def create_video(
        queue: Queue
):
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    queue : Queue
        The queue to get the generated images from.
    """
    print(queue.qsize())
    global final_images
    while not queue.empty():
        resulting_img = postprocess_image(queue.get(block=False), output_type="pil")[0]
        final_images.append(resulting_img)
        print('Final Images: {}'.format(len(final_images)))

def main(
    #model_id_or_path: str = "KBlueLeaf/kohaku-v2.1",
    model_id_or_path: str ="stabilityai/sd-turbo",
    lora_dict: Optional[Dict[str, float]] = None,
    #prompt: str = "1girl with brown dog hair, thick glasses, smiling",
    #prompt: str = "1girl, anime, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details",
    #prompt: str = "landscape, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details",
    #prompt: str = "picasso style",
    #prompt: str = "anime",
    prompt: str = "painting picasso, full of color, detailed",
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

    # First, process the video frames.
    #screen()
    #input_screen = threading.Thread(target=screen, args=())
    #input_screen.start()

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
            similar_image_filter_max_skip_frame
            ),
    )
    process1.start()
    
    '''image_generation_process(queue,
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
    )'''

    #process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
    #process2.start()
    

    # terminate
    #process2.join()
    #print("process2 terminated.")
    #close_queue.put(True)
    #print("process1 terminating...")
    process1.join(3) # with timeout
    #if process1.is_alive():
    #    print("process1 still alive. force killing...")
    #    process1.terminate() # force kill...
    #process1.join()
    print("process1 terminated.")

    # Now, we process the video
    create_video(queue)


if __name__ == "__main__":
    fire.Fire(main)