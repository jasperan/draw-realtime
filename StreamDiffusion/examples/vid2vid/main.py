import os
import sys
from typing import Literal, Dict, Optional
import cv2
import fire
import torch
from torchvision.io import read_video, write_video
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

import argparse

cli_parser = argparse.ArgumentParser()
cli_parser.add_argument('-p', '--path', type=str,
                    default="""H:/github/draw-realtime/StreamDiffusion/examples/vid2vid/video""",
                    required=False,
                    help='Path to the input video directory')

args = cli_parser.parse_args()
input_path = args.path



def main(
    input: str = "/home/ubuntu/videos/video.mp4",
    output: str = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "output.mp4"),
    #model_id: str = "KBlueLeaf/kohaku-v2.1",
    model_id: str = "stabilityai/sd-turbo",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "painting picasso, full of color, detailed",
    scale: float = 1.0,
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
    use_denoising_batch: bool = True,
    enable_similar_image_filter: bool = True,
    seed: int = 2,
):

    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    input_path : str, optional
        The input video path directory to load images from.
    output : str, optional
        The output video name to save images to.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    scale : float, optional
        The scale of the image, by default 1.0.
    acceleration : Literal["none", "xformers", "tensorrt"]
        The type of acceleration to use for image generation.
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    enable_similar_image_filter : bool, optional
        Whether to enable similar image filter or not,
        by default True.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    """

    #directory_path = "/home/ubuntu/videos"
    #dir_path = """H:\github\draw-realtime\StreamDiffusion\examples\vid2vid\video"""
    #global input_path
    #dir_path = input_path
    #file_list = os.listdir(dir_path)
    #file_paths = [os.path.join(dir_path, file) for file in file_list]

    #print(file_paths)


    #video_info = read_video(input_path)
    video_info = read_video(input)
    video = video_info[0] / 255
    fps = video_info[2]["video_fps"]
    width = int(video.shape[1] * scale)
    height = int(video.shape[2] * scale)

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id,
        lora_dict=lora_dict,
        t_index_list=[35, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        use_tiny_vae=True, # TAESD
        warmup=10,
        acceleration=acceleration,
        do_add_noise=False,
        mode="img2img",
        output_type="pt",
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=0.98,
        use_denoising_batch=use_denoising_batch,
        seed=seed,
    )

    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    video_result = torch.zeros(video.shape[0], width, height, 3)

    for _ in range(stream.batch_size):
        stream(image=video[0].permute(2, 0, 1))

    for i in tqdm(range(video.shape[0])):
        output_image = stream(video[i].permute(2, 0, 1))
        video_result[i] = output_image.permute(2, 1, 0)

    video_result = video_result * 255
    write_video(output, video_result[2:], fps=fps)


if __name__ == "__main__":
    fire.Fire(main)
