import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import json
import multiprocessing
from functools import partial
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from utils import load_caption, save_json, load_desc, load_json,\
    join_captions, load_sub, prepare_sub, join_subtitles, merge_captions_segment_wise
from parse_data import clean_subtitles
from prompts import STAGE2TASK, SINGLE_SCENE_TASK_PROMPT, MULTI_SCENE_TASK_PROMPT, FULL_SCENE_TASK_PROMPT
from monitor_process import send_email
from openai import OpenAI
import time
import random


STAGE = 'full'
MODEL_PATH = "Qwen/Qwen2.5-72B-Instruct"
BATCH_SIZE = 48
SYSTEM_PROMPT = "You are a helpful assistant that generates questions and answers based on audio and video captions."
API_KEY = "sk-d4099bd527ba48ba9d0fa6e58b35bfff"
MODEL_NAME = "qwen2.5-72b-instruct"
client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=API_KEY, 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
def process_openai_api( messages, max_retries=5):
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
    caption_path = 'paired_captions_v1.json' 
    desc_path = 'raw_data/descriptions.json'
    sub_path = './raw_data/subtitle'
    save_path = './results'
    
    paired_captions = load_caption(caption_path)
    vid2covid = load_json('./vid2covid.json')
    vid2desc = load_desc(desc_path)
    vid2sub = load_sub(sub_path)
    
    vid_names = list(paired_captions.keys())
    
    # vid_names = ['Y_b5wYLVmyw', 'VzK8Ed4IMBk', '_FkBmrmnELU']
    


    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        dtype='bfloat16',
    )
    sampling_params = SamplingParams(
        temperature=0.1, 
        top_p=0.001, 
        repetition_penalty=1.05, 
        max_tokens=1024,
    )

    if STAGE == 'single':  
        scene_level = STAGE2TASK[STAGE]
        for task in scene_level:
            # if task not in ['task8']:
                # continue
            qas = list()
            all_messages = []
            aud_forms = scene_level[task]
            for vid in tqdm(vid_names, desc = f'Processing [{task}]'):
                if vid not in vid2covid:
                    continue
                covid = vid2covid[vid]
                vid_captions = paired_captions[vid]
                subtitles = clean_subtitles(vid2sub[vid])
                vid_captions = prepare_sub(subtitles, vid_captions)
                texts = list()
                metadatas = list()
                vid_captions = [i for i in vid_captions if i['segment_id'] in covid]
                for caption in vid_captions:
                    segment_id, start_time, end_time = caption['segment_id'], caption['start_time'], caption['end_time']
                    prompt_completion = dict()
                    if 'speech' in aud_forms:
                        if caption['subtitle'] == '':
                            continue
                    if 'speech_emotion' in aud_forms:
                        if caption['speech_emotion'] == '':
                            continue
                    if 'sound_event' in aud_forms:
                        if caption['sound_event'] == '':
                            continue
                    if 'music' in aud_forms:
                        if caption['music'] == '':
                            continue
                    segments_info= merge_captions_segment_wise(vid_captions, subtitles, aud_forms, segment_id = segment_id)
                    prompt_completion['segments_info'] = segments_info

                    task_prompt = SINGLE_SCENE_TASK_PROMPT(task).format(**prompt_completion)
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": task_prompt}
                    ]   
                    metadata = (vid,segment_id, start_time, end_time)
                    # all_messages.append((messages, metadata))
                    # completion = client.chat.completions.create(
                    #     model=MODEL_NAME,
                    #     messages=messages,
                    #     temperature=0.1, 
                    #     top_p=0.001, 
                    #     presence_penalty =1.05, 
                    #     max_tokens=1024,
                    #     )
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    metadatas.append((segment_id, start_time, end_time))
                    texts.append(text)
                responses = list()
                for i in range(0,len(texts), BATCH_SIZE):
                    responses += llm.generate(texts[i:i+BATCH_SIZE], sampling_params=sampling_params, use_tqdm = False)
                for (segment_id, start_time, end_time), response in zip(metadatas, responses):
                    generated_text = response.outputs[0].text
                    # generated_text = completion.choices[0].message.content
            # with multiprocessing.Pool(processes=64) as pool:
            #     # 使用partial固定client参数
            #     results = list(tqdm(
            #         pool.imap(process_openai_api, all_messages),
            #         total=len(all_messages),
            #         desc=f'Processing API calls [{task}]'
            #     ))
                
                # 将结果与视频信息匹配
                # for metadata, generated_text in results:
                    if generated_text is None:
                        continue
                    vid, segment_id, start_time, end_time = metadata
                    generated_caption = {
                        'video_name':vid,
                        'task':task,
                        'segment_id': segment_id,
                        'start_time': start_time,
                        'end_time': end_time,
                        'caption': generated_text,
                    }
                    qas.append(generated_caption)

            save_json(qas, os.path.join(save_path, f'{task}_v1.json'))
            # save_json(qas, os.path.join(save_path, f'{task}_prompting.json'))
    if STAGE == 'multi':  
        scene_level = STAGE2TASK[STAGE]
        for task in scene_level:
            # if task not in ['task12']:
            #     continue
            qas = list()
            # if os.path.exists(os.path.join(save_path, f'{task}_v1.json')):
            #     qas = load_json(os.path.join(save_path, f'{task}_v1.json'))
            #     vid_names = [i['video_name'] for i in qas if i['caption'] is None]
            all_messages = []
            aud_forms = scene_level[task]
            for vid in tqdm(vid_names, desc = f'Processing [{task}]'):
                if vid not in vid2covid:
                    continue
                covid = vid2covid[vid]
                video_captions = paired_captions[vid]
                video_captions = [i for i in video_captions if i['segment_id'] in covid]
                subtitles = clean_subtitles(vid2sub[vid])
                vid_caption, music_caption, sound_caption, speech_emotion = join_captions(video_captions)
                joined_sub = join_subtitles(subtitles)
                prompt_completion = dict()
                # prompt_completion['video_caption'] = vid_caption
                if 'speech' in aud_forms:
                    if joined_sub == '':
                        continue
                    elif len(subtitles)<=8:
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
                segments_info= merge_captions_segment_wise(video_captions, subtitles, aud_forms)
                prompt_completion['segments_info'] = segments_info
                
                task_prompt = MULTI_SCENE_TASK_PROMPT(task).format(**prompt_completion)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task_prompt}
                ]   
            #     all_messages.append((messages, vid))

            # with multiprocessing.Pool(processes=24) as pool:
            #     # 使用partial固定client参数
            #     results = list(tqdm(
            #         pool.imap(process_openai_api, all_messages),
            #         total=len(all_messages),
            #         desc=f'Processing API calls [{task}]'
            #     ))
                
            #     # 将结果与视频信息匹配
            #     for video_name, generated_text in results:
            #         caption = {
            #             'video_name': video_name,
            #             'task': task,
            #             'caption': generated_text,
            #         }
            #         qas.append(caption)

                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                tokens = tokenizer.encode(text)
                if len(tokens) >= 30000:
                    generated_text = None
                else:
                    try:
                        response = llm.generate([text], sampling_params=sampling_params, use_tqdm = False)
                    except RuntimeError as e:
                        print(e)
                    generated_text = response[0].outputs[0].text
                
                caption ={
                    'video_name':vid,
                    'task':task,
                    'caption': generated_text,
                }
                qas.append(caption)
            save_json(qas, os.path.join(save_path, f'{task}_v1.json'))
    if STAGE == 'full':  
        scene_level = STAGE2TASK[STAGE]
        for task in scene_level:
            # if task not in ['task17']:
            #     continue
            qas = list()
            all_messages = []
            aud_forms = scene_level[task]
            for vid in tqdm(vid_names, desc = f'Processing [{task}]'):
                if vid not in vid2desc:
                    continue
                vid_desc = vid2desc[vid]
                video_captions = paired_captions[vid]
                subtitles = clean_subtitles(vid2sub[vid])
                vid_caption, music_caption, sound_caption, speech_emotion = join_captions(video_captions)
                joined_sub = join_subtitles(subtitles)
                prompt_completion = dict()
                # prompt_completion['video_caption'] = vid_caption
                # prompt_completion['video_description'] = vid_desc
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
                segments_info= merge_captions_segment_wise(video_captions, subtitles, aud_forms, vid_desc=vid_desc)
                prompt_completion['segments_info'] = segments_info
                
                task_prompt = FULL_SCENE_TASK_PROMPT(task).format(**prompt_completion)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task_prompt}
                ]   
                # all_messages.append((messages, vid))

            # with multiprocessing.Pool(processes=24) as pool:
            #     # 使用partial固定client参数
            #     results = list(tqdm(
            #         pool.imap(process_openai_api, all_messages),
            #         total=len(all_messages),
            #         desc=f'Processing API calls [{task}]'
            #     ))
                
            #     # 将结果与视频信息匹配
            #     for video_name, generated_text in results:
            #         caption = {
            #             'video_name': video_name,
            #             'task': task,
            #             'caption': generated_text,
            #         }
            #         qas.append(caption)
                # completion = client.chat.completions.create(
                #     model=MODEL_NAME,
                #     messages=messages,
                #     temperature=0.1, 
                #     top_p=0.001, 
                #     presence_penalty =1.05, 
                #     max_tokens=1024,
                # )
                # generated_text = completion.choices[0].message.content
                
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                tokens = tokenizer.encode(text)
                if len(tokens) >= 30000:
                    generated_text = None
                else:
                    try:
                        response = llm.generate([text], sampling_params=sampling_params, use_tqdm = False)
                    except RuntimeError as e:
                        print(e)
                        
                generated_text = response[0].outputs[0].text
                caption ={
                    'video_name':vid,
                    'task':task,
                    'caption': generated_text,
                }
                qas.append(caption)
            save_json(qas, os.path.join(save_path, f'{task}_v1.json'))
            
            
    # send_email(
    #     subject='程序结束通知',
    #     body=f'程序已经结束，请及时查看！'
    # )