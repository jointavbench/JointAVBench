# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
warnings.filterwarnings("ignore")

def load_video(video_path, max_frames_num,fps=1,force_sample=False, start_time=None, end_time=None):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    # 计算起始和终止帧索引
    if start_time is not None:
        start_frame_idx = int(start_time * vr.get_avg_fps())
    else:
        start_frame_idx = 0
    total_frame_num = len(vr)
    
    if end_time is not None:
        end_frame_idx = int(end_time * vr.get_avg_fps())
    else:
        end_frame_idx = total_frame_num
    video_time = (end_frame_idx-start_frame_idx) / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(start_frame_idx, end_frame_idx - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

def load_model(device="cuda"):
    pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
    model_name = "llava_qwen"
    device_map = {"": device}
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    model.eval().cuda(device)
    return tokenizer, model, image_processor, max_length
    
    
def evaluation(input_data, modality, model_conf, device):
    tokenizer, model, image_processor, max_length = model_conf
    prompt, file_path, segments, _ = input_data
    video_path = file_path+'.mp4'
    max_frames_num = 32
    video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda(device).bfloat16()
    video = [video]
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n{prompt}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    cont = model.generate(
        input_ids,
        images=video,
        modalities= ["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=256,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

    return text_outputs