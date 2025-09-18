import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
import json
from collections import defaultdict
from tqdm import tqdm
import pickle
import copy
import time
import random
from openai import OpenAI
import multiprocessing
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from utils import load_caption, load_json, save_json, load_desc, load_sub, prepare_sub, join_subtitles, merge_captions_segment_wise, join_captions
from prompts import INTERVAL_CHECK_PROMPT, STAGE2TASK
from parse_data import filter_qa_general, clean_subtitles, extract_segment_interval

# MODEL_PATH = "Qwen/Qwen2.5-72B-Instruct"
MODEL_PATH = "/data-01/jianghan/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31"
STAGE = 'multi'
API_KEY = "sk-d4099bd527ba48ba9d0fa6e58b35bfff"
MODEL_NAME = "qwen2.5-72b-instruct"
# Define the system prompt
SYSTEM_PROMPT = """
Your task is to identify the segment intervals based on the information used in the question-answer pair. Please follow the instructions to identify the interval.
"""
client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=API_KEY, 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
def process_openai_api(messages, max_retries=5):
    """处理单个API调用的函数"""
    message, metadata = messages
    video_name = metadata[0]
    time.sleep(random.randint(1,3))
    retry_count = 0
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=message,
                temperature=0.1, 
                top_p=0.001, 
                presence_penalty=1.05, 
                max_tokens=1024,
            )
            return metadata, completion.choices[0].message.content
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                # 指数退避等待时间 (2, 4, 8, 16秒) + 随机抖动
                wait_time = min(2 ** retry_count + random.uniform(0, 1), 30)  # 最大不超过30秒
                print(f"视频 {video_name} 第 {retry_count} 次尝试失败，{str(e)}，等待 {wait_time:.1f} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"视频 {video_name} 达到最大重试次数 {max_retries} 仍失败，错误: {str(e)}")
                return metadata, None

if __name__ == '__main__':
    caption_path = 'paired_captions.json' 
    sub_path = './raw_data/subtitle'
    save_path = './results'
    paired_captions = load_caption(caption_path)
    
    vid2sub = load_sub(sub_path)
    vid2covid = load_json('./vid2covid.json')
    vid_names = list(paired_captions.keys())
    # task2qa = defaultdict(list)
    # filtered_qas = filter_qa_general(save_path, STAGE, True)
    # for qa in filtered_qas:
    #     task2qa[qa['task']].append(qa)
    qa_path = './task2qa_general.json'
    task2qa = load_json(qa_path)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # llm = LLM(
    #     model=MODEL_PATH,
    #     tensor_parallel_size=4,
    #     gpu_memory_utilization=0.9,
    #     dtype='bfloat16',
    # )
    # sampling_params = SamplingParams(
    #     temperature=0.1, 
    #     top_p=0.001, 
    #     repetition_penalty=1.05, 
    #     max_tokens=1024,
    # )

    scene_level = STAGE2TASK[STAGE]
    for task in scene_level:
        if task not in ['task9', 'task13']:
            continue
        qas = list()
        aud_forms = scene_level[task]
        generated_qas = task2qa[task]
        vid2qa = defaultdict(list)
        if os.path.exists(os.path.join(save_path, f'{task}_intervals.json')):
            generated_qas = load_json(os.path.join(save_path, f'{task}_intervals.json'))
            for item in generated_qas:
                if item['segment_id'] is not None:
                    qas.append(item)
                    continue
                video_name = item['video_name']
                vid2qa[video_name].append(item)
        else:
            for item in generated_qas:
                video_name = item['video_name']
                # segment_id = {k:v for k,v in item if k != 'video_name'}
                vid2qa[video_name].append(item)
        vid_names = list(vid2qa.keys())
        vid_names.sort()
        for vid in tqdm(vid_names, desc = f'Processing [{task}]'):
            covid = vid2covid[vid]
            video_captions = paired_captions[vid]
            video_captions = [i for i in video_captions if i['segment_id'] in covid]
            subtitles = clean_subtitles(vid2sub[vid])
            joined_sub = join_subtitles(subtitles)
            vid_caption, music_caption, sound_caption, speech_emotion = join_captions(video_captions)
            if 'speech' in aud_forms:
                if joined_sub == '':
                    continue
            if 'speech_emotion' in aud_forms:
                if speech_emotion == '':
                    continue
            if 'sound_event' in aud_forms:
                if sound_caption == '':
                    continue
            if 'music' in aud_forms:
                if music_caption == '':
                    continue
            prompt_completion = dict()
            segments_info= merge_captions_segment_wise(video_captions, subtitles, aud_forms)
            prompt_completion['segments_info'] = segments_info
            available_qas = vid2qa[vid]
            for available_qa in available_qas:
                prompt_completion['question'] = available_qa['question']
                prompt_completion['answer'] = available_qa['answer']
                prompt_completion['explanation'] = available_qa['explanation']
                
                task_prompt = INTERVAL_CHECK_PROMPT.format(**prompt_completion)
                message = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task_prompt}
                ]   
                _, generated_text = process_openai_api([message, [vid]])
                # text = tokenizer.apply_chat_template(
                #     message,
                #     tokenize=False,
                #     add_generation_prompt=True
                # )
                # tokens = tokenizer.encode(text)
                # if len(tokens) >= 30000:
                #     generated_text = ''
                # else:
                #     try:
                #         response = llm.generate([text], sampling_params=sampling_params, use_tqdm = False)
                #         generated_text = response[0].outputs[0].text
                #     except RuntimeError as e:
                #         tokens = tokenizer.encode(text)
                #         print(len(tokens))
                #         print(e)
                
                segment_info = extract_segment_interval(generated_text)
                if segment_info is None:
                    intervals = None
                else:
                    intervals = [segment_info['start'],segment_info['end']]
                caption ={
                    'qid':available_qa['qid'],
                    'video_name':vid,
                    'task':task,
                    'segment_id':intervals,
                    'question':available_qa['question'],
                    'answer':available_qa['answer'],
                    'explanation':available_qa['explanation'],
                    'caption': generated_text,
                }
                qas.append(caption)
        save_json(qas, os.path.join(save_path, f'{task}_intervals.json'))

        