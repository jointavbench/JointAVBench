import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
import json
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from utils import load_caption, load_json, save_json, load_desc, load_sub, prepare_sub, join_subtitles, join_captions
from prompts import QA_MODALITY_JUDGE_PROMPT, QA_QUALITY_JUDGE_PROMPT, STAGE2TASK
from parse_data import parse_single_scene, parse_multi_scene, clean_subtitles
from openai import OpenAI
import multiprocessing
import argparse
import time
import random

MODEL_PATH = "Qwen/Qwen2.5-72B-Instruct"
BATCH_SIZE = 32
# STAGE = 'multi'
# JUDGE_STAGE = 'modality'
# Get API key from environment variable
API_KEY = os.environ.get('QWEN_API_KEY', '')
MODEL_NAME = "qwen2.5-72b-instruct"
judge_system_prompt = {
    'modality':"""
You are a multimodal assistant designed to analyze whether questions about video content can be answered using video text, audio text, or both. Follow the provided instructions to determine if a question requires one or both modalities.
""",
    'quality':"""
You are a quality evaluation assistant. Your task is to assess the quality of a question-answer pair by checking its format and content. Follow the provided instructions to determine if a question-answer pair passes quality check.
"""
}

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', help='Stage of processing (single, multi, full)', required=False, default='single')
    parser.add_argument('--judge-stage', help='Stage of judging (modality, quality)', required=False, default='quality')
    args = parser.parse_args()
    STAGE = args.stage
    JUDGE_STAGE = args.judge_stage
    SYSTEM_PROMPT = judge_system_prompt[JUDGE_STAGE]
    QA_JUDGE_PROMPT = QA_MODALITY_JUDGE_PROMPT if JUDGE_STAGE =='modality' else QA_QUALITY_JUDGE_PROMPT
    save_path = './results'
    
    # vid_names = ['Y_b5wYLVmyw', 'VzK8Ed4IMBk', '_FkBmrmnELU']
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # llm = LLM(
    #     model=MODEL_PATH,
    #     tensor_parallel_size=4,
    #     gpu_memory_utilization=0.95,
    #     dtype='bfloat16',
    # )
    # sampling_params = SamplingParams(
    #     temperature=0.1, 
    #     top_p=0.001, 
    #     repetition_penalty=1.05, 
    #     max_tokens=1024,
    # )
    if STAGE == 'single':  
        scene_level = STAGE2TASK[STAGE]
        for task in scene_level:
            # if task not in ['task2']:
                # continue
            if task not in ['task8']:
                continue
            qa_judges = list()
            generated_qas = parse_single_scene(os.path.join(save_path,task+'.json'))
            # print(task, len(generated_qas))
            # continue
            # for qa_idx in tqdm(range(0, len(generated_qas), BATCH_SIZE), desc = f'Processing [{task}]'):
            #     texts = list()
            #     metadata = list()
            #     for available_qa in generated_qas[qa_idx:qa_idx+BATCH_SIZE]:
            #         prompt_completion = dict()
            #         prompt_completion['question'] = available_qa['question']
            #         prompt_completion['answer'] = available_qa['answer']
            #         prompt_completion['explanation'] = available_qa['explanation']
                    
            #         task_prompt = QA_JUDGE_PROMPT.format(**prompt_completion)
            #         messages = [
            #             {"role": "system", "content": SYSTEM_PROMPT},
            #             {"role": "user", "content": task_prompt}
            #         ]   
            #         text = tokenizer.apply_chat_template(
            #             messages,
            #             tokenize=False,
            #             add_generation_prompt=True
            #         )
            #         metadata.append((available_qa['video_name'], available_qa['segment_id'], available_qa['start_time'], available_qa['end_time']))
            #         texts.append(text)
            #     responses = llm.generate(texts, sampling_params=sampling_params, use_tqdm = False)
            #     for (vid, segment_id, start_time, end_time), response in zip(metadata, responses):
            #         generated_text = response.outputs[0].text
            #         caption ={
            #             'video_name':vid,
            #             'task':task,
            #             'segment_id': segment_id,
            #             'start_time': start_time,
            #             'end_time': end_time,
            #             'judgement_text': generated_text,
            #             'judgement': 'YES' if '[YES]' in generated_text else 'NO',
            #         }
            #         qa_judges.append(caption)
            all_messages = list()
            # generated_qas = [i for i in generated_qas if i['video_name'] in vid_names][:20]
            for available_qa in generated_qas:
                metadata=(available_qa['video_name'], available_qa['segment_id'], available_qa['start_time'], available_qa['end_time'])
                prompt_completion = dict()
                prompt_completion['question'] = available_qa['question']
                prompt_completion['answer'] = available_qa['answer']
                prompt_completion['explanation'] = available_qa['explanation']
                
                task_prompt = QA_JUDGE_PROMPT.format(**prompt_completion)
                message = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task_prompt}
                ]   
                all_messages.append((message, metadata))
                
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
                        'video_name':vid,
                        'task':task,
                        'segment_id': segment_id,
                        'start_time': start_time,
                        'end_time': end_time,
                        'judgement_text': generated_text,
                        'judgement': 'YES' if (generated_text is not None and '[YES]' in generated_text) else 'NO',
                    }
                    qa_judges.append(caption)
            save_json(qa_judges, os.path.join(save_path, f'{task}_{JUDGE_STAGE}_judgements.json'))
    if STAGE == 'multi' or STAGE == 'full':  
        scene_level = STAGE2TASK[STAGE]
        for task in scene_level:
            if task in ['task9']:
                continue
            qa_judges = list()
            generated_qas = parse_multi_scene(os.path.join(save_path,task+'.json'))
            # print(task, len(generated_qas))
            # continue
            # for qa_idx in tqdm(range(0, len(generated_qas), BATCH_SIZE), desc = f'Processing [{task}]'):
            #     texts = list()
            #     metadata = list()
            #     for available_qa in generated_qas[qa_idx:qa_idx+BATCH_SIZE]:
            #         prompt_completion = dict()
            #         prompt_completion['question'] = available_qa['question'].strip()
            #         prompt_completion['answer'] = available_qa['answer'].strip()
            #         prompt_completion['explanation'] = available_qa['explanation'].strip()
            #         if prompt_completion['question'] == '' or prompt_completion['answer'] == '' or prompt_completion['explanation'] == '':
            #             continue
            #         task_prompt = QA_JUDGE_PROMPT.format(**prompt_completion)
            #         messages = [
            #             {"role": "system", "content": SYSTEM_PROMPT},
            #             {"role": "user", "content": task_prompt}
            #         ]   
            #         text = tokenizer.apply_chat_template(
            #             messages,
            #             tokenize=False,
            #             add_generation_prompt=True
            #         )
            #         metadata.append((available_qa['video_name'],available_qa['question']))
            #         texts.append(text)
            #     responses = llm.generate(texts, sampling_params=sampling_params, use_tqdm = False)
            #     for (vid, question), response in zip(metadata, responses):
            #         generated_text = response.outputs[0].text
            #         caption ={
            #             'video_name':vid,
            #             'task':task,
            #             'question':question,
            #             'judgement_text': generated_text,
            #             'judgement': 'YES' if '[YES]' in generated_text else 'NO',
            #         }
            #         qa_judges.append(caption)
            all_messages = list()
            # generated_qas = [i for i in generated_qas if i['video_name'] in vid_names]
            for available_qa in generated_qas:
                metadata=(available_qa['video_name'],available_qa['question'])
                prompt_completion = dict()
                prompt_completion['question'] = available_qa['question'].strip()
                prompt_completion['answer'] = available_qa['answer'].strip()
                prompt_completion['explanation'] = available_qa['explanation'].strip()
                if prompt_completion['question'] == '' or prompt_completion['answer'] == '' or prompt_completion['explanation'] == '':
                    continue
                task_prompt = QA_JUDGE_PROMPT.format(**prompt_completion)
                message = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task_prompt}
                ]   
                all_messages.append((message, metadata))
                
            with multiprocessing.Pool(processes=64) as pool:
                # 使用partial固定client参数
                results = list(tqdm(
                    pool.imap(process_openai_api, all_messages),
                    total=len(all_messages),
                    desc=f'Processing API calls [{task}]'
                ))
                
                # 将结果与视频信息匹配
                for metadata, generated_text in results:
                    vid, question = metadata
                    caption ={
                        'video_name':vid,
                        'task':task,
                        'question':question,
                        'judgement_text': generated_text,
                        'judgement': 'YES' if (generated_text is not None and '[YES]' in generated_text) else 'NO',
                    }
                    qa_judges.append(caption)
            save_json(qa_judges, os.path.join(save_path, f'{task}_{JUDGE_STAGE}_judgements.json'))       
