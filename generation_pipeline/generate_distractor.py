import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
import json
import pickle
from tqdm import tqdm
import argparse
from collections import defaultdict
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from utils import load_caption, load_json, save_json, load_desc, load_sub, prepare_sub, join_subtitles, join_captions, merge_captions_segment_wise
from prompts import distractor_generation_prompt, STAGE2TASK
from parse_data import filter_qa_general, clean_subtitles, extract_distractors, update_segment_interval
from openai import OpenAI
import multiprocessing
import time
import random

MODEL_PATH = "Qwen/Qwen2.5-72B-Instruct"
BATCH_SIZE = 32

# Define the system prompt
SYSTEM_PROMPT = """
Your task is to generate three plausible but incorrect distractors for video-based multiple-choice questions. Use visual and audio details from the video segment to create distractors that align with the content but are factually wrong. Follow the provided instructions to generate distractors.
"""
API_KEY = "sk-d4099bd527ba48ba9d0fa6e58b35bfff"
MODEL_NAME = "qwen2.5-72b-instruct"
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
                return metadata, ''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', help='Stage of processing (single, multi, full)', required=True)
    args = parser.parse_args()
    stage = args.stage
    caption_path = 'paired_captions.json' 
    desc_path = 'raw_data/descriptions.json'
    sub_path = './raw_data/subtitle'
    save_path = './results'
    paired_captions = load_caption(caption_path)
    
    vid2desc = load_desc(desc_path)
    vid2sub = load_sub(sub_path)
    task2qa = load_json('./task2qa_specific.json')
    vid2covid = load_json('./vid2covid.json')
    # test_vid_names = ['Y_b5wYLVmyw', 'VzK8Ed4IMBk', '_FkBmrmnELU']
    test_vid_names = load_json('./video_names.json')
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

    if stage == 'single':  
        scene_level = STAGE2TASK[stage]
        for task in scene_level:
            # if task not in ['task8']:
            #     continue
            distractors = list()
            aud_forms = scene_level[task]
            generated_qas = task2qa[task]
            vid2qa = defaultdict(list)
            all_messages = list()

            for item in generated_qas:
                video_name = item['video_name']
                # segment_id = {k:v for k,v in item if k != 'video_name'}
                vid2qa[video_name].append(item)
            vid_names = list(vid2qa.keys())
            vid_names = [i for i in vid_names if i not in test_vid_names]
            for vid in tqdm(vid_names, desc = f'Processing [{task}]'):
                video_captions = paired_captions[vid]
                subtitles = clean_subtitles(vid2sub[vid])
                covid = vid2covid[vid]
                video_captions = [i for i in video_captions if i['segment_id'] in covid]
                texts = list()
                metadatas = list()
                available_qas = vid2qa[vid]
                for available_qa in available_qas:
                    qid = available_qa['qid']
                    segment_id = available_qa['segment_id']
                    prompt_completion = dict()
                    prompt_completion['question'] = available_qa['question']
                    prompt_completion['answer'] = available_qa['answer']
                    prompt_completion['explanation'] = available_qa['explanation']

                    segments_info = merge_captions_segment_wise(video_captions, subtitles, aud_forms, segment_id = segment_id)
                    prompt_completion['segments_info'] = segments_info
                    task_prompt = distractor_generation_prompt.format(**prompt_completion)
                    message = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": task_prompt}
                    ]   
                    metadata = (vid, qid, segment_id, available_qa['question'], available_qa['answer'], available_qa['explanation'])
                    all_messages.append((message, metadata))
                    text = tokenizer.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    metadatas.append(metadata)
                    texts.append(text)
                responses = list()
                for i in range(0,len(texts), BATCH_SIZE):
                    responses += llm.generate(texts[i:i+BATCH_SIZE], sampling_params=sampling_params, use_tqdm = False)
                for (vid, qid, segment_id, question, answer, explanation), response in zip(metadatas, responses):
                    generated_text = response.outputs[0].text
                    caption ={
                        'qid':qid,
                        'video_name':vid,
                        'task':task,
                        'segment_id': segment_id,
                        'distractors_text': generated_text,
                        'question': question,
                        'correct_answer': answer,
                        'explanation':explanation,
                        'options':extract_distractors(generated_text),
                    }
                    distractors.append(caption)
            # with multiprocessing.Pool(processes=64) as pool:
            #     # 使用partial固定client参数
            #     results = list(tqdm(
            #         pool.imap(process_openai_api, all_messages),
            #         total=len(all_messages),
            #         desc=f'Processing API calls [{task}]'
            #     ))
            #     for metadata, generated_text in results:
            #         vid, qid, segment_id, question, answer, explanation = metadata
            #         caption ={
            #             'qid':qid,
            #             'video_name':vid,
            #             'task':task,
            #             'segment_id': segment_id,
            #             'distractors_text': generated_text,
            #             'question': question,
            #             'correct_answer': answer,
            #             'explanation':explanation
            #             'options':extract_distractors(generated_text),
            #         }
            #         distractors.append(caption)
                    
            save_json(distractors, os.path.join(save_path, f'{task}_distractors.json'))
            # save_json(distractors, os.path.join('prompt_result', f'{task}_distractors_v1.json'))
    if stage == 'multi':  
        scene_level = STAGE2TASK[stage]
        for task in scene_level:
            distractors = list()
            aud_forms = scene_level[task]
            generated_qas = task2qa[task]
            vid2qa = defaultdict(list)
            if os.path.exists(os.path.join(save_path, f'{task}_distractors_all.json')):
                generated_qas = load_json(os.path.join(save_path, f'{task}_distractors_all.json'))
                for item in generated_qas:
                    if item['distractors_text'] != '':
                        distractors.append(item)
                        continue
                    video_name = item['video_name']
                    vid2qa[video_name].append(item)
            else:
                for item in generated_qas:
                    video_name = item['video_name']
                    # segment_id = {k:v for k,v in item if k != 'video_name'}
                    vid2qa[video_name].append(item)
            vid_names = list(vid2qa.keys())
            vid_names = [i for i in vid_names if i not in test_vid_names]
            for vid in tqdm(vid_names, desc = f'Processing [{task}]'):
                video_captions = paired_captions[vid]
                subtitles = clean_subtitles(vid2sub[vid])
                covid = vid2covid[vid]
                video_captions = [i for i in video_captions if i['segment_id'] in covid]
                
                available_qas = vid2qa[vid]
                for available_qa in available_qas:
                    prompt_completion = dict()
                    segment_id = available_qa['segment_id']
                    segments_info= merge_captions_segment_wise(video_captions, subtitles, aud_forms, segment_id = segment_id)
                    prompt_completion['segments_info'] = segments_info
                    prompt_completion['question'] = available_qa['question']
                    prompt_completion['answer'] = available_qa['correct_answer']
                    prompt_completion['explanation'] = available_qa['explanation']
                    task_prompt = distractor_generation_prompt.format(**prompt_completion)
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
                    #         generated_text = ''
                    caption ={
                        'qid':available_qa['qid'],
                        'video_name':vid,
                        'task':task,
                        'segment_id': segment_id,
                        'distractors_text': generated_text,
                        'question': available_qa['question'],
                        'correct_answer': available_qa['correct_answer'],
                        'explanation': available_qa['explanation'],
                        'options':extract_distractors(generated_text),
                    }
                    distractors.append(caption)

            save_json(distractors, os.path.join(save_path, f'{task}_distractors_all.json'))
    if stage == 'full':  
        scene_level = STAGE2TASK[stage]
        for task in scene_level:
            # if task in ['task15', 'task16']:
            #     continue
            distractors = list()
            aud_forms = scene_level[task]
            generated_qas = task2qa[task]
            vid2qa = defaultdict(list)
            if os.path.exists(os.path.join(save_path, f'{task}_distractors_all.json')):
                generated_qas = load_json(os.path.join(save_path, f'{task}_distractors_all.json'))
                for item in generated_qas:
                    if item['distractors_text'] != '':
                        distractors.append(item)
                        continue
                    video_name = item['video_name']
                    vid2qa[video_name].append(item)
            else:
                for item in generated_qas:
                    video_name = item['video_name']
                    # segment_id = {k:v for k,v in item if k != 'video_name'}
                    vid2qa[video_name].append(item)
            vid_names = list(vid2qa.keys())
            vid_names = [i for i in vid_names if i not in test_vid_names]
            for vid in tqdm(vid_names, desc = f'Processing [{task}]'):
                vid_desc = vid2desc[vid]
                video_captions = paired_captions[vid]
                subtitles = clean_subtitles(vid2sub[vid])
                prompt_completion = dict()
                segments_info= merge_captions_segment_wise(video_captions, subtitles, aud_forms, vid_desc = vid_desc)
                prompt_completion['segments_info'] = segments_info
                
                available_qas = vid2qa[vid]
                for available_qa in available_qas:
                    prompt_completion['question'] = available_qa['question']
                    prompt_completion['answer'] = available_qa['correct_answer']
                    prompt_completion['explanation'] = available_qa['explanation']
                    
                    task_prompt = distractor_generation_prompt.format(**prompt_completion)
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
                    #         generated_text = ''
                    #         print(len(tokens))
                    #         print(e)
                    caption ={
                        'qid':available_qa['qid'],
                        'video_name':vid,
                        'task':task,
                        'segment_id': None,
                        'distractors_text': generated_text,
                        'question': available_qa['question'],
                        'correct_answer': available_qa['correct_answer'],
                        'explanation': available_qa['explanation'],
                        'options':extract_distractors(generated_text),
                    }
                    distractors.append(caption)

            save_json(distractors, os.path.join(save_path, f'{task}_distractors_all.json'))