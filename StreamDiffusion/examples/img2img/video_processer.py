import os
import sys
from typing import Literal, Dict, Optional
import cv2
from PIL import Image
import fire
import moviepy.video.io.ImageSequenceClip
import warnings
warnings.filterwarnings("ignore")


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

count: int = 0

def create_video(image_folder: str):
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    queue : Queue
        The queue to get the generated images from.
    """
    fps=30

    image_files = [os.path.join(image_folder,img)
                for img in os.listdir(image_folder)
                if img.endswith(".jpg")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile('./output/my_video.mp4')


def screen(path):
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    global count
    count = 0

    print('Video Properties: {}x{}'.format(image.shape[1], image.shape[0]))    
    while success:
        cv2.imwrite("./tmp/frames/frame_%d.jpg" % count, image)
        #print(type(image))
        #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
        success, image = vidcap.read()
        count += 1
        #print(type(pil_img))
        #inputs.append(pil2tensor(pil_img))
        print('Frame {}'.format(count))

def main(
    input: str = os.path.join(CURRENT_DIR, "..", "..", "images", "inputs", "input.png"),
    output: str = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "output.png"),
    model_id_or_path: str = "stabilityai/sd-turbo",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "",
    negative_prompt: str = "low quality, bad quality, blurry, low resolution",
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
    use_denoising_batch: bool = True,
    guidance_scale: float = 1.2,
    cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    seed: int = 2,
    delta: float = 0.5,
):
    """
    Initializes the StreamDiffusionWrapper.

    Parameters
    ----------
    input : str, optional
        The input image file to load images from.
    output : str, optional
        The output image file to save images to.
    model_id_or_path : str
        The model id or path to load.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    negative_prompt : str, optional
        The negative prompt to use.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"], optional
        The acceleration method, by default "tensorrt".
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    guidance_scale : float, optional
        The CFG scale, by default 1.2.
    cfg_type : Literal["none", "full", "self", "initialize"],
    optional
        The cfg_type for img2img mode, by default "self".
        You cannot use anything other than "none" for txt2img mode.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    delta : float, optional
        The delta multiplier of virtual residual noise,
        by default 1.0.
    """

    global count
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='Path to the video file', required=True,
                        default='../vid2vid/video/video.mp4')
    args = parser.parse_args()
    screen(args.path) # process video first

    if guidance_scale <= 1.0:
        cfg_type = "none"

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[22, 32, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=False,
        enable_similar_image_filter=True,
        similar_image_filter_threshold=.99,
        similar_image_filter_max_skip_frame=10,
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
    frame_builder = 0
    for x in range(count):
        file_name = './tmp/frames/frame_{}.jpg'.format(x)
        image_tensor = stream.preprocess_image(file_name)

        for _ in range(stream.batch_size - 1):
            stream(image=image_tensor)
        
        output_image = stream(image=image_tensor)
        #print(type(output_image))

        output_image.save('./tmp/processed/frame_{}.jpg'.format(x))

    create_video("./tmp/processed")
    


if __name__ == "__main__":
    fire.Fire(main)
