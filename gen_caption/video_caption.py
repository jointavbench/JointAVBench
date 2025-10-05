import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'
from scenedetect import detect, ContentDetector, open_video, SceneManager
import cv2
import json
import pickle
import torch
from transformers import AutoProcessor
from vision_process import process_vision_info
from vllm import LLM, SamplingParams
from datetime import datetime, timedelta
from tqdm import tqdm

from torchvision import transforms
from PIL import Image
from utils import PySceneSegmenter, seconds2timestamp, timestamp2seconds, load_jsonl, load_json


SYSTEM_PROMPT = """
You are a video captioning assistant. Your task is to generate detailed and accurate captions for movie video clips based on the user's input. Use clear and vivid language to ensure the caption reflects the content of the video.
"""

CAPTION_PROMPT = """
Please generate a clear, concise, and detailed caption for the provided movie video clip. Ensure the description is entirely based on the visual content of the video, without any speculation, assumptions, or uncertain information. Focus on capturing the most essential visual elements while avoiding unnecessary repetition or overly verbose language. Describe the scene as follows:
1. Scene Setting: Briefly describe the environment, location, time of day, lighting, and any notable objects or elements in the background. Only include details that are clearly visible in the video.
2. Characters and Actions: Highlight the appearance, clothing, and key actions of any characters present. Focus on their most significant movements, gestures, and interactions. Do not infer emotions, intentions, or backstory unless explicitly shown through visual cues.
3. Scene Dynamics: Describe any important changes in the scene. Include the sequence of events and the pacing of the scene to convey how it unfolds over time. Only describe what is visually evident.
4. Emotional Tone: Convey the mood or atmosphere through the most impactful visual cues, such as facial expressions, body language, or environmental details. Only describe emotions that are clearly expressed through visible actions or expressions.
5. Key Events: Highlight any significant events or actions that occur within the scene, focusing on their narrative importance or impact on the characters. Only include events that are explicitly shown in the video.
Important Notes:
- Strictly factual: Ensure the description is entirely based on the visual content of the video. Do not include any information that is not explicitly shown or clearly visible.
- Avoid speculation: Do not infer or assume anything about characters' thoughts, feelings, or motivations unless they are directly expressed through visible actions or expressions.
- Acknowledge uncertainty: If certain details are unclear or ambiguous, do not include them in the description. Focus only on what is clearly visible.
- Exclude text: If there are subtitles or text displayed at the bottom of the video, do not include them in the caption. Focus only on the visual elements of the scene.
- Be concise: Avoid unnecessary details or repetition. Focus on the most critical visual elements that define the scene.
Use vivid but economical language to paint a clear and engaging picture of the scene for someone who cannot see the video.
"""


CAPTION_SAVE_PATH = 'prompt_result/prompt_v7.jsonl'


if __name__ == "__main__":
        
    MODEL_PATH = "Qwen/Qwen2.5-VL-72B-Instruct"
    # MODEL_PATH = "/fs/fast/u2021201666/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-72B-Instruct/snapshots/71dfcaa79ed29adcc6afb4419a4f9e8cc74af05a/"

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.95,
        limit_mm_per_prompt={ "video": 1},
        dtype='bfloat16',
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        min_tokens = 10,
        max_tokens = 2048,
        stop_token_ids=[],
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type":"video",
                    "video":'',
                    "fps": 1.0,
                    "segments":'',
                    "min_pixels": 64 * 28 * 28,
                    "max_pixels": 512 * 28 * 28,
                    "max_frames": 128,
                },
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        },
    ]
    # For video input, you can pass following values instead:
    # "type": "video",
    # "video": "<video URL>",
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    with open('./vid2datasetpath.pkl', 'rb') as f:
        vid2datasetpath = pickle.load(f)
    # with open('./vids_names.pkl', 'rb') as f:
    #     vids_names = pickle.load(f)
    # with open('./en_vids.pkl', 'rb') as f:
    #     en_vids = pickle.load(f)
    vid2coverage = load_json('./coverage.json')
    sample_names = list(vid2coverage.keys())
    # testset_dir = '/fs/fast/u2021201666/samples'
    testset_dir = '/fs/fast/u2021201666/caption_data'
    # sample_names = os.listdir(testset_dir)
    # sample_names = vids_names['main']
    # unprocessed_sample_names = list()
    # for sample_name in sample_names:
    #     sample_folder = os.path.join(testset_dir, sample_name)
    #     # if sample_name not in en_vids:
    #     #     continue
    #     if not os.path.exists(sample_folder):
    #         continue
    #     if not os.path.exists(os.path.join(sample_folder, sample_name+'_video_caption_coverage.jsonl')):
    #         unprocessed_sample_names.append(sample_name)
    # sample_names = unprocessed_sample_names
    # response_list = list()
    # sample_names = ['sf4Kcse5KPk']
    for vid_idx, video_name in tqdm(enumerate(sample_names), total = len(sample_names[520:])):
        sample_dir = os.path.join(testset_dir, video_name)
        # vid_path = [os.path.join(sample_dir, i) for i in os.listdir(sample_dir) if i.endswith('mp4')][0]
        vid_path = os.path.join(vid2datasetpath[video_name], video_name+'.mp4')
        segment_path = [os.path.join(sample_dir, i) for i in os.listdir(sample_dir) if i.endswith('pkl')][0]
        caption_path = os.path.join(sample_dir, video_name+'_video_caption_coverage.jsonl')
        # caption_path = CAPTION_SAVE_PATH
        if os.path.exists(caption_path):
            continue
        segmenter = PySceneSegmenter(save_path = segment_path)
        segments_timestamp = [[start_time, end_time] for start_time, end_time in segmenter]
        # segments = [[timestamp2seconds(start_time), timestamp2seconds(end_time)] for start_time, end_time in segments_timestamp]
        
        
        vid_coverage = vid2coverage[video_name]
        if not vid_coverage:
            continue
        segments = list()
        segments_timestamp_new = list()
        for covered_segment in vid_coverage:
            start_time = segments_timestamp[covered_segment[0]][0]
            end_time = segments_timestamp[covered_segment[-1]][-1]
            segments_timestamp_new.append([start_time, end_time])
            segments.append([timestamp2seconds(start_time), timestamp2seconds(end_time)])

        # segment_ids = list()
        # if not os.path.exists(caption_path):
        #     segment_ids = list(range(len(segments)))
        # else:
        #     caption_data = load_jsonl(caption_path)
        #     caption_idx = [item['segment_id'] for item in caption_data if item['caption'] != ""]
        #     for idx in range(len(segments)):
        #         if idx not in caption_idx:
        #             segment_ids.append(idx)

        #     if len(segment_ids) == 0:
        #         continue
        # segment_ids.sort()
        # segments = [segments[i] for i in segment_ids]
        
        messages[-1]['content'][0]['video'] = vid_path
        messages[-1]['content'][0]['segments'] = segments
        image_inputs, video_inputs = process_vision_info(messages)
        # List[Tensor(120, c, h, w)]
        # to_pil = transforms.ToPILImage()
        # for idx, video_input in tqdm(enumerate(video_inputs[0]), total = len(video_inputs[0]), desc = f'Captioning {video_name} [{vid_idx+1}/{len(sample_names)}]'):
        for idx, video_input in enumerate(video_inputs[0]):
            # video_input = [image for image in video_input]
            # for image_idx, image_tensor in enumerate(video_input):
            #     image_pil = to_pil(image_tensor.to(torch.uint8))
            #     image_pil.save(f"./imgs/segment{idx}_img{image_idx}.jpg")
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_input is not None:
                mm_data["video"] = video_input

            llm_input = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            }
            
            outputs = llm.generate([llm_input], sampling_params=sampling_params, use_tqdm = False)
            generated_text = outputs[0].outputs[0].text
                
            caption ={
                    'video_name':video_name,
                    'type':'video_caption',
                    'segment_id': idx,
                    'start_time':segments_timestamp_new[idx][0],
                    'end_time':segments_timestamp_new[idx][1],
                    'caption':generated_text,
                }
            # response_list.append(caption)
            # print('generated_text', generated_text)
            try:
                with open(caption_path, 'a', encoding='utf-8') as f:
                    json.dump(caption, f, ensure_ascii=False)
                    f.write('\n')
            except Exception as e:
                print(f"An error occurred while writing to the file: {e}")