import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
import json
import re
import pickle
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from utils import load_caption, load_json, save_json, load_desc, load_sub, prepare_sub, join_subtitles, join_captions, merge_captions_segment_wise
from prompts import sequence_check_prompt_single, sequence_check_prompt_multi, speech_emotion_check_prompt, sound_check_prompt,\
    music_check_prompt, interval_check_prompt, STAGE2TASK, task4_ambiguity_check_prompt, task2_ambiguity_check_prompt
from parse_data import parse_single_scene, parse_multi_scene, clean_subtitles
import argparse
from openai import OpenAI
import multiprocessing
import argparse
import time
import random

BATCH_SIZE = 48
SYSTEM_PROMPT = """
You are an expert in filtering QA pairs.
""".strip()
MODEL_PATH = "Qwen/Qwen2.5-72B-Instruct"
# Get API key from environment variable
API_KEY = os.environ.get('QWEN_API_KEY', '')
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

# Define the system prompt
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', help='', choices = ['sequence', 'stage', 'audio', 'ambiguity'], required=True)
    args = parser.parse_args()
    caption_path = 'paired_captions.json' 
    desc_path = 'raw_data/descriptions.json'
    sub_path = './raw_data/subtitle'
    save_path = './results'
    qa_path = './task2qa_interval.json'
    task2qa = load_json(qa_path)
    paired_captions = load_caption(caption_path)
    test_vid_names = ['Y_b5wYLVmyw', 'VzK8Ed4IMBk', '_FkBmrmnELU']
    vid2desc = load_desc(desc_path)
    vid2sub = load_sub(sub_path)
    vid2covid = load_json('./vid2covid.json')
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # llm = LLM(
    #     model=MODEL_PATH,
    #     tensor_parallel_size=8,
    #     gpu_memory_utilization=0.8,
    #     dtype='bfloat16',
    # )
    # sampling_params = SamplingParams(
    #     temperature=0.1, 
    #     top_p=0.001, 
    #     repetition_penalty=1.05, 
    #     max_tokens=1024,
    # )
    if args.check == 'stage':
        for stage in ['single', 'multi', 'full']:
            scene_level = STAGE2TASK[stage]
            if stage == 'single':
                user_prompt = qa_material_judge_universal_prompt
            elif stage == 'multi':
                user_prompt = qa_correct_check_prompt
            elif stage == 'full':
                user_prompt = qa_commonsense_check_prompt
            else:
                assert 1==0
            for task in scene_level:
                if task in ['task3']:
                    continue
                qa_judges = list()
                aud_forms = scene_level[task]
                generated_qas = task2qa[task]
                vid2qa = defaultdict(list)
                all_messages = list()
                for item in generated_qas:
                    video_name = item['video_name']
                    # segment_id = {k:v for k,v in item if k != 'video_name'}
                    vid2qa[video_name].append(item)
                vid_names = list(vid2qa.keys())
                vid_names = [i for i in vid_names if i in test_vid_names]
                for vid in tqdm(vid_names, desc = f'Processing [{task}]'):
                    vid_captions = paired_captions[vid]
                    subtitles = clean_subtitles(vid2sub[vid])
                    if stage in ['single', 'multi']:
                        covid = vid2covid[vid]
                        vid_captions = [i for i in vid_captions if i['segment_id'] in covid]
                    elif stage in ['full']:
                        vid_desc = vid2desc[vid]
                    texts = list()
                    metadata = list()
                    available_qas = vid2qa[vid]
                    for available_qa in available_qas:
                        prompt_completion = dict()
                        prompt_completion['question'] = available_qa['question']
                        prompt_completion['answer'] = available_qa['answer']
                        prompt_completion['explanation'] = available_qa['explanation']
                        
                        if task == 'single':
                            segment_id = available_qa['segment_id']
                            segments_info = merge_captions_segment_wise(vid_captions, subtitles, aud_forms, segment_id = segment_id)
                            prompt_completion['segments_info'] = segments_info
                        elif task == 'multi':
                            segments_info= merge_captions_segment_wise(vid_captions, subtitles, aud_forms)
                            prompt_completion['segments_info'] = segments_info
                        else:
                            segments_info= merge_captions_segment_wise(vid_captions, subtitles, aud_forms, vid_desc=vid_desc)
                            prompt_completion['segments_info'] = segments_info
                        message = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt.format(**prompt_completion)}
                        ]   
                        metadata = (vid, available_qa['qid'], available_qa['segment_id'])
                        all_messages.append((message, metadata))
                    #     text = tokenizer.apply_chat_template(
                    #         messages,
                    #         tokenize=False,
                    #         add_generation_prompt=True
                    #     )
                    #     metadata.append((available_qa['qid'], available_qa['segment_id']))
                    #     texts.append(text)
                    # responses = list()
                    # for i in range(0,len(texts), BATCH_SIZE):
                    #     responses += llm.generate(texts[i:i+BATCH_SIZE], sampling_params=sampling_params, use_tqdm = False)
                    # for (qid, segment_id), response in zip(metadata, responses):
                    #     generated_text = response.outputs[0].text
                    #     caption ={
                    #         'qid':qid,
                    #         'video_name':vid,
                    #         'task':task,
                    #         'segment_id': segment_id,
                    #         'judgement_text': generated_text,
                    #         'judgement': 'YES' if '[YES]' in generated_text else 'NO',
                    #     }
                    #     qa_judges.append(caption)
                with multiprocessing.Pool(processes=64) as pool:
                    # 使用partial固定client参数
                    results = list(tqdm(
                        pool.imap(process_openai_api, all_messages),
                        total=len(all_messages),
                        desc=f'Processing API calls [{task}]'
                    ))
                    
                    # 将结果与视频信息匹配
                    for metadata, generated_text in results:
                        vid, segment_id, start_time, end_time = metadata
                        caption ={
                            'qid':qid,
                            'video_name':vid,
                            'task':task,
                            'segment_id': segment_id,
                            'judgement_text': generated_text,
                            'judgement': 'YES' if '[YES]' in generated_text else 'NO',
                        }
                        qa_judges.append(caption)

                save_json(qa_judges, os.path.join(save_path, f'{task}_{args.check}.json'))
    elif args.check == 'sequence':
        sequence_prompt = {'task8':sequence_check_prompt_single, 'task10':sequence_check_prompt_multi}
        for task in ['task8','task10']:
            if task == 'task8':
                continue
            user_prompt = sequence_prompt[task]
            qa_judges = list()
            all_messages = list()
            aud_forms = STAGE2TASK['single']['task8'] if task == 'task8' else STAGE2TASK['multi']['task10']
            generated_qas = task2qa[task]
            vid2qa = defaultdict(list)
            all_messages = list()
            for item in generated_qas:
                video_name = item['video_name']
                vid2qa[video_name].append(item)
            vid_names = list(vid2qa.keys())
            # vid_names = [i for i in vid_names if i in test_vid_names]
            for vid in tqdm(vid_names, desc = f'Processing [{task}]'):
                vid_captions = paired_captions[vid]
                subtitles = clean_subtitles(vid2sub[vid])
                covid = vid2covid[vid]
                vid_captions = [i for i in vid_captions if i['segment_id'] in covid]
                available_qas = vid2qa[vid]
                for available_qa in available_qas:
                    prompt_completion = dict()
                    prompt_completion['question'] = available_qa['question']
                    prompt_completion['answer'] = available_qa['answer']
                    prompt_completion['explanation'] = available_qa['explanation']
                    
                    segment_id = available_qa['segment_id']
                    if task == 'task10':
                        min_segment, max_segment = min(segment_id), max(segment_id)
                        if max_segment-min_segment<2:
                            continue
                    segments_info = merge_captions_segment_wise(vid_captions, subtitles, aud_forms, segment_id = segment_id)
                    prompt_completion['segments_info'] = segments_info
                    message = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt.format(**prompt_completion)}
                    ]   
                    # message = tokenizer.apply_chat_template(
                    #     message,
                    #     tokenize=False,
                    #     add_generation_prompt=True
                    # )
                    metadata = (vid, available_qa['qid'], available_qa['segment_id'])
                    all_messages.append((message, metadata))

            # results = list()
            # for i in tqdm(range(0,len(all_messages), BATCH_SIZE)):
            #     texts = [messages[0] for messages in all_messages[i:i+BATCH_SIZE]]
            #     metadatas = [messages[1] for messages in all_messages[i:i+BATCH_SIZE]]
            #     responses = llm.generate(texts, sampling_params=sampling_params, use_tqdm = False)
            #     responses = [response.outputs[0].text for response in responses]
            #     results += list(zip(metadatas, responses))
            with multiprocessing.Pool(processes=24) as pool:
                # 使用partial固定client参数
                results = list(tqdm(
                    pool.imap(process_openai_api, all_messages),
                    total=len(all_messages),
                    desc=f'Processing API calls [{task}]'
                ))
                
            # 将结果与视频信息匹配
            for metadata, generated_text in results:
                vid, qid, segment_id= metadata
                caption ={
                    'qid':qid,
                    'video_name':vid,
                    'task':task,
                    'segment_id': segment_id,
                    'judgement_text': generated_text,
                    'judgement': 'YES' if '[YES]' in generated_text else 'NO',
                    # 'segment_info':segment_info,
                    # 'question':q,
                    # 'answer':a,
                    # 'explanation':e,
                }
                qa_judges.append(caption)
            save_json(qa_judges, os.path.join(save_path, f'{task}_{args.check}.json'))
    elif args.check == 'audio':
        audio_prompt = {
            "speech_emotion":speech_emotion_check_prompt,
            "sound_event": sound_check_prompt,
            "music": music_check_prompt,
        }
        available_tasks = {
            'speech':[],
            'speech_emotion':['task2', 'task6'],
            'sound_event':['task4', 'task5'],
            'music':['task7'],
        }
        for audio_type in audio_prompt:
            if audio_type != 'music':
                continue
            user_prompt = audio_prompt[audio_type]
            for stage in ['single', 'multi', 'full']:
                scene_level = STAGE2TASK[stage]
                for task in scene_level:
                    if task not in available_tasks[audio_type]:
                        continue
                    qa_judges = list()
                    aud_forms = scene_level[task]
                    generated_qas = task2qa[task]
                    vid2qa = defaultdict(list)
                    all_messages = list()
                    for item in generated_qas:
                        video_name = item['video_name']
                        # segment_id = {k:v for k,v in item if k != 'video_name'}
                        vid2qa[video_name].append(item)
                    vid_names = list(vid2qa.keys())
                    # vid_names = [i for i in vid_names if i in test_vid_names]
                    for vid in tqdm(vid_names, desc = f'Processing [{task}]'):
                        vid_captions = paired_captions[vid]
                        subtitles = clean_subtitles(vid2sub[vid])
                        if stage in ['single', 'multi']:
                            covid = vid2covid[vid]
                            vid_captions = [i for i in vid_captions if i['segment_id'] in covid]
                        elif stage in ['full']:
                            vid_desc = vid2desc[vid]
                        texts = list()
                        metadatas = list()
                        available_qas = vid2qa[vid]
                        for available_qa in available_qas:
                            prompt_completion = dict()
                            prompt_completion['question'] = available_qa['question']
                            prompt_completion['answer'] = available_qa['answer']
                            prompt_completion['explanation'] = available_qa['explanation']
                            
                            if stage == 'single':
                                segment_id = available_qa['segment_id']
                                segments_info = merge_captions_segment_wise(vid_captions, subtitles, aud_forms, segment_id = segment_id)
                                prompt_completion['segments_info'] = segments_info
                            elif stage == 'multi':
                                segments_info= merge_captions_segment_wise(vid_captions, subtitles, aud_forms)
                                prompt_completion['segments_info'] = segments_info
                            else:
                                segments_info= merge_captions_segment_wise(vid_captions, subtitles, aud_forms, vid_desc=vid_desc)
                                prompt_completion['segments_info'] = segments_info
                            message = [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt.format(**prompt_completion)}
                            ]   
                            metadata = (vid, available_qa['qid'], available_qa['segment_id'], segments_info, available_qa['question'], available_qa['answer'], available_qa['explanation'])
                            all_messages.append((message, metadata))
                        #     text = tokenizer.apply_chat_template(
                        #         messages,
                        #         tokenize=False,
                        #         add_generation_prompt=True
                        #     )
                        #     metadatas.append(metadata)
                        #     texts.append(text)
                        # responses = list()
                        # for i in range(0,len(texts), BATCH_SIZE):
                        #     responses += llm.generate(texts[i:i+BATCH_SIZE], sampling_params=sampling_params, use_tqdm = False)
                        # for (vid, qid, segment_id), response in zip(metadata, responses):
                        #     generated_text = response.outputs[0].text
                        #     caption ={
                        #         'qid':qid,
                        #         'video_name':vid,
                        #         'task':task,
                        #         'segment_id': segment_id,
                        #         'judgement_text': generated_text,
                        #         'judgement': 'YES' if '[YES]' in generated_text else 'NO',
                        #     }
                        #     qa_judges.append(caption)
                    with multiprocessing.Pool(processes=48) as pool:
                        # 使用partial固定client参数
                        results = list(tqdm(
                            pool.imap(process_openai_api, all_messages),
                            total=len(all_messages),
                            desc=f'Processing API calls [{task}]'
                        ))
                        
                        # 将结果与视频信息匹配
                        for metadata, generated_text in results:
                            vid, qid, segment_id, segments_info, q, a, e= metadata
                            caption ={
                                'qid':qid,
                                'video_name':vid,
                                'task':task,
                                'segment_id': segment_id,
                                'judgement_text': generated_text,
                                'judgement': 'YES' if '[YES]' in generated_text else 'NO',
                                # 'segments_info':segments_info,
                                # 'question':q,
                                # 'answer':a,
                                # 'explanation':e,
                            }
                            qa_judges.append(caption)

                    save_json(qa_judges, os.path.join(save_path, f'{task}_{args.check}_{audio_type}.json'))
    elif args.check == 'ambiguity':
        task2prompt = {
            'task2':task2_ambiguity_check_prompt,
            'task4':task4_ambiguity_check_prompt,
            'task5':task4_ambiguity_check_prompt
            }
        scene_level = STAGE2TASK['single']
        for task in ['task2', 'task4', 'task5']:
            if task in ['task2']:
                continue
            qa_judges = list()
            user_prompt = task2prompt[task]
            aud_forms = scene_level[task]
            generated_qas = task2qa[task]
            vid2qa = defaultdict(list)
            all_messages = list()
            for item in generated_qas:
                video_name = item['video_name']
                # segment_id = {k:v for k,v in item if k != 'video_name'}
                vid2qa[video_name].append(item)
            vid_names = list(vid2qa.keys())
            # vid_names = [i for i in vid_names if i in test_vid_names]
            for vid in tqdm(vid_names, desc = f'Processing [{task}]'):
                vid_captions = paired_captions[vid]
                subtitles = clean_subtitles(vid2sub[vid])
                covid = vid2covid[vid]
                vid_captions = [i for i in vid_captions if i['segment_id'] in covid]

                texts = list()
                metadatas = list()
                available_qas = vid2qa[vid]
                for available_qa in available_qas:
                    prompt_completion = dict()
                    prompt_completion['question'] = available_qa['question']
                    prompt_completion['answer'] = available_qa['answer']
                    prompt_completion['explanation'] = available_qa['explanation']
                    
                    segment_id = available_qa['segment_id']
                    segments_info = merge_captions_segment_wise(vid_captions, subtitles, aud_forms, segment_id = segment_id)
                    prompt_completion['segments_info'] = segments_info

                    message = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt.format(**prompt_completion)}
                    ]   
                    metadata = (vid, available_qa['qid'], available_qa['segment_id'], segments_info, available_qa['question'],available_qa['answer'],available_qa['explanation'])
                    all_messages.append((message, metadata))
                #     text = tokenizer.apply_chat_template(
                #         messages,
                #         tokenize=False,
                #         add_generation_prompt=True
                #     )
                #     metadatas.append(metadata)
                #     texts.append(text)
                # responses = list()
                # for i in range(0,len(texts), BATCH_SIZE):
                #     responses += llm.generate(texts[i:i+BATCH_SIZE], sampling_params=sampling_params, use_tqdm = False)
                # for (vid, qid, segment_id), response in zip(metadata, responses):
                #     generated_text = response.outputs[0].text
                #     caption ={
                #         'qid':qid,
                #         'video_name':vid,
                #         'task':task,
                #         'segment_id': segment_id,
                #         'judgement_text': generated_text,
                #         'judgement': 'YES' if '[YES]' in generated_text else 'NO',
                #     }
                #     qa_judges.append(caption)
            with multiprocessing.Pool(processes=48) as pool:
                # 使用partial固定client参数
                results = list(tqdm(
                    pool.imap(process_openai_api, all_messages),
                    total=len(all_messages),
                    desc=f'Processing API calls [{task}]'
                ))
                
                # 将结果与视频信息匹配
                for metadata, generated_text in results:
                    vid, qid, segment_id, segments_info, q, a, e= metadata
                    caption ={
                        'qid':qid,
                        'video_name':vid,
                        'task':task,
                        'segment_id': segment_id,
                        'judgement_text': generated_text,
                        'judgement': 'YES' if '[YES]' in generated_text else 'NO',
                        # 'segments_info':segments_info,
                        # 'question':q,
                        # 'answer':a,
                        # 'explanation':e,
                    }
                    qa_judges.append(caption)

            save_json(qa_judges, os.path.join(save_path, f'{task}_{args.check}.json'))
