import os
os.environ['DECORD_EOF_RETRY_MAX'] = '40960'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import re
import json
import shutil
import random
import time
import pickle
import torch
import fcntl
import argparse
import multiprocessing
from functools import partial
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from utils import load_json, save_json, load_jsonl, timestamp2seconds, task2audform, merge_captions_segment_wise, clean_subtitles, load_caption, load_sub
from tqdm import tqdm


def extract_model_answer(
    model_output: str,
    prefix2text: Dict[str, str],
    possible_prefixes: List[str] = None,
    answer_patterns: List[str] = None
) -> Optional[str]:
    """
    从模型输出中提取答案，支持多种匹配方式：
    1. 使用正则表达式匹配常见答案模式
    2. 直接查找选项前缀
    3. 查找输出中是否包含选项对应的文本内容
    
    参数:
        model_output: 模型的原始输出文本
        prefix2text: 选项前缀到选项内容的映射字典，如 {'A': 'Paris', 'B': 'London'}
        possible_prefixes: 可能的选项前缀列表，默认为['A', 'B', 'C', 'D']
        answer_patterns: 用于匹配答案的正则表达式模式列表
        
    返回:
        匹配到的单个选项前缀(如'A')，如果匹配到多个选项或没有匹配到则返回None
    """
    # 设置默认参数
    if possible_prefixes is None:
        possible_prefixes = ['A', 'B', 'C', 'D']
    
    if answer_patterns is None:
        answer_patterns = [
            r'answer(?: is)?\s*[:=-]?\s*(["\'(]?\s*[A-D]\s*["\')]?)',  # 匹配 "Answer: A", "answer is 'B'", "answer is (C)", "answer is "D""
            r'([A-D])(?:\s*is the|\)?\s*is)\s*(?:correct|right|answer)',  # "A is correct" 或 "B) is right" 或 "C is the answer"
            r'option\s*([A-D])\b',  # "option A" 或 "choose option C"
            r'select(?:ed)?\s*([A-D])\b',  # "select B" 或 "selected D"
            r'["\'(]\s*([A-D])\s*["\')]\s*(?:is|as)\s*(?:the )?(?:answer|correct)',  # 匹配 "'A' is the answer", "(B) is correct", '"C" as answer'
        ]
    
    # 预处理：移除换行和多余空格
    cleaned_text = ' '.join(model_output.split())
    
    # 1. 尝试所有正则匹配模式
    found_prefixes = set()
    for pattern in answer_patterns:
        matches = re.finditer(pattern, cleaned_text, re.IGNORECASE)
        for match in matches:
            extracted = match.group(1).upper()
            if extracted in possible_prefixes:
                found_prefixes.add(extracted)
    
    # 2. 尝试直接查找选项前缀
    for prefix in possible_prefixes:
        if re.search(rf'\b{prefix}\b(?=\.\s)', cleaned_text):
            found_prefixes.add(prefix.upper())
    
    # 4. 如果前几种方法都失败，检查是否包含选项文本内容
    for prefix, text in prefix2text.items():
        # 确保prefix在可能的选项中
        if prefix.upper() in possible_prefixes:
            # 检查选项文本是否在模型输出中
            if text in cleaned_text or text.lower() in cleaned_text:
                found_prefixes.add(prefix.upper())
    
    # 5. 对于非常短的输出，尝试更宽松的匹配
    if len(cleaned_text.split()) == 1:
        for prefix in possible_prefixes:
            if prefix in cleaned_text.strip().upper():
                found_prefixes.add(prefix)
        
    # 3. 检查是否包含多个选项
    if len(found_prefixes) > 1:
        return None
    
    # 如果已经找到一个明确的选项，直接返回
    if len(found_prefixes) == 1:
        return found_prefixes.pop()
    return None

def generate_mcq_prompt(
    question: str,
    options: List[str],
    correct_answer: str,
    option_prefixes: List[str] = None
) -> Tuple[str, str]:
    """
    生成多选题的prompt文本并记录正确答案标记
    
    参数:
        question: 问题文本
        options: 选项文本列表
        correct_answer: 正确答案文本
        option_prefixes: 可选的选项前缀列表(如['A', 'B', 'C'])
        
    返回:
        tuple: (生成的prompt文本, 正确答案前缀如"A")
    """
    prefix2text = {}
    # 设置默认选项前缀(A., B., C., D.)
    if option_prefixes is None:
        option_prefixes = ['A', 'B', 'C', 'D']
    
    # 验证输入
    if len(options) < 2:
        raise ValueError("必须提供至少2个选项")
    if len(options) > len(option_prefixes):
        raise ValueError(f"选项数量({len(options)})超过前缀数量({len(option_prefixes)})")
    if correct_answer not in options:
        raise ValueError("正确答案不在提供的选项中")
    
    # 打乱选项顺序(但记录原始索引)
    indexed_options = list(enumerate(options))
    random.shuffle(indexed_options)
    
    # 构建带前缀的选项文本和正确答案跟踪
    prefixed_options = []
    correct_prefix = None
    
    for idx, (original_idx, option) in enumerate(indexed_options):
        prefix = option_prefixes[idx]
        prefixed_option = f"{prefix}. {option}"
        prefixed_options.append(prefixed_option)
        prefix2text[prefix] = option
        # 检查是否是正确答案
        if options[original_idx] == correct_answer:
            correct_prefix = prefix
    
    # 拼接问题和选项
    options_text = "\n".join(prefixed_options)
    prompt = f"{question}\n{options_text}"
    
    return prompt, correct_prefix, prefix2text

def get_segment_time(segment_id, segments_list):
    if isinstance(segment_id, int):
        start_segment, end_segment = segment_id, segment_id
    elif isinstance(segment_id, list):
        start_segment, end_segment = min(segment_id), max(segment_id)
    else:  # None
        start_segment, end_segment = 0, -1
    start_time = segments_list[start_segment][0]
    end_time = segments_list[end_segment][-1]
    return [start_time, end_time]

def batch_inputs(inputs, metadata, max_duration=300):
    current_batch = []  
    current_metadata = []
    current_duration = 0 
    for input_data, meta in zip(inputs,metadata):
        start_time, end_time= input_data[2]
        duration = end_time-start_time
        if current_duration + duration <= max_duration:
            current_batch.append(input_data)
            current_metadata.append(meta)
            current_duration += duration
        else:
            if current_batch:
                yield current_batch, current_metadata
            current_batch = [input_data] 
            current_metadata = [meta] 
            current_duration = duration

    if current_batch:
        yield current_batch, current_metadata
        
def batch_inputs_num(inputs, metadata, max_num=5):
    current_batch = []  
    current_metadata = []
    for input_data, meta in zip(inputs,metadata):
        if len(current_batch) < max_num:
            current_batch.append(input_data)
            current_metadata.append(meta)
        else:
            yield current_batch, current_metadata
            current_batch = [input_data] 
            current_metadata = [meta] 

    if current_batch:
        yield current_batch, current_metadata

def eval_batch(inputs, metadata, evaluation, modality, model_name, save_path):
    results = list()
    pbar = tqdm(total = len(inputs))
    for input_batch, input_meta in batch_inputs_num(inputs, metadata):
        # print(input_meta[0]['qid'])
        texts = evaluation(input_batch, modality)
        for idx, text in enumerate(texts):
            model_answer = extract_model_answer(text, input_batch[idx][3])
            model_result = {
                **input_meta[idx],
                'modality':modality,
                'model':model_name,
                'model_answer':model_answer,
                'model_output':text,
            }
            with open(save_path, 'a', encoding='utf-8') as f:
                json.dump(model_result, f, ensure_ascii=False)
                f.write('\n')
            results.append(model_result)
        pbar.update(len(texts))
    return results

def shuffle_data(inputs, metadata):
    """Shuffle inputs & metadata together to maintain alignment."""
    combined = list(zip(inputs, metadata))
    random.shuffle(combined)  # In-place shuffle
    shuffled_inputs, shuffled_metadata = zip(*combined)
    return list(shuffled_inputs), list(shuffled_metadata)

def chunk_data(inputs, metadata, num_chunks):
    """Split inputs & metadata into N chunks for parallel processing."""
    chunk_size = (len(inputs) + num_chunks - 1) // num_chunks  # Round up division
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        yield inputs[start:end], metadata[start:end]
        
def process_chunk(task, eval_func, modality, model_name, save_path):
    """Process a chunk of data on the assigned GPU."""
    chunk_inputs, chunk_metadata, device = task
    # Load model on the assigned GPU
    import time
    import random
    time.sleep(random.randint(1, 10))
    model_conf = load_model_for_device(model_name, device)
    results = []
    for input_data, meta_data in tqdm(zip(chunk_inputs, chunk_metadata), total=len(chunk_inputs), desc=f'Processing on {device}'):
        text = eval_func(input_data, modality, model_conf=model_conf, device = device)
        model_answer = extract_model_answer(text, input_data[3])
        model_result = {
            **meta_data,
            'modality': modality,
            'model': model_name,
            'model_answer': model_answer,
            'model_output': text,
        }
        
        # Thread-safe file writing (optional: use file lock if needed)
        with open(save_path, 'a', encoding='utf-8') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(model_result, f, ensure_ascii=False)
            f.write('\n')
            fcntl.flock(f, fcntl.LOCK_UN)
        
        results.append(model_result)
    
    return results

def eval_multiprocess(inputs, metadata, eval_func, modality, model_name, save_path):
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    num_gpus = torch.cuda.device_count()
    devices = [f'cuda:{i}' for i in range(num_gpus)] if num_gpus > 0 else ['cpu']

    # Shuffle data (while keeping inputs & metadata aligned)
    shuffled_inputs, shuffled_metadata = shuffle_data(inputs, metadata)
    
    # Split data into N chunks (N = number of GPUs)
    data_chunks = list(chunk_data(shuffled_inputs, shuffled_metadata, len(devices)))
    
    # Assign each chunk to a GPU
    tasks = []
    for i, (chunk_inputs, chunk_metadata) in enumerate(data_chunks):
        device = devices[i % len(devices)]
        tasks.append((chunk_inputs, chunk_metadata, device))
    
    # Process each chunk in parallel
    with multiprocessing.Pool(processes=len(devices)) as pool:
        func = partial(process_chunk, eval_func=eval_func, modality=modality, model_name=model_name, save_path=save_path)
        results = list(pool.imap_unordered(func, tasks))
    
    return results

def load_model_for_device(model_name, device):
    if model_name == 'videollama2':
        from eval_videollama2 import load_model
        return load_model(device)
    elif model_name == 'omnir1':
        from eval_omnir1 import load_model
        return load_model(device)
    elif model_name == 'videollama3':
        from eval_videollama3 import load_model
        return load_model(device)
    elif model_name == 'vita1.5':
        from eval_vita import load_model
        return load_model(device)
    elif model_name == 'llavavideo':
        from eval_llavavideo import load_model
        return load_model(device)
    elif model_name == 'internvl':
        from eval_internvl import load_model
        return load_model(device)
    elif model_name == 'kimiaudio':
        from eval_kimi import load_model
        return load_model(device)
    elif model_name == 'avicuna':
        from eval_avicuna import load_model
        return load_model(device)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

def process_chunk_api(task, eval_func, modality, model_name, save_path):
    """Process a chunk of data on the assigned GPU."""
    input_data, meta_data = task
    # Load model on the assigned GPU
    results = []
    text = eval_func(input_data, modality)
    if text == None:
        return None
    model_answer = extract_model_answer(text, input_data[3])
    model_result = {
        **meta_data,
        'modality': modality,
        'model': model_name,
        'model_answer': model_answer,
        'model_output': text,
    }
    time.sleep(random.uniform(0, 1))
    
    # Thread-safe file writing (optional: use file lock if needed)
    with open(save_path, 'a', encoding='utf-8') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(model_result, f, ensure_ascii=False)
        f.write('\n')
        fcntl.flock(f, fcntl.LOCK_UN)
    return model_result

def eval_api(inputs, metadata, eval_func, modality, model_name, save_path):
    avail_qids = load_json('./avail_qids.json')
    parallel_num = 3
    eval_data = list(zip(inputs, metadata))
    eval_data = [(input_item, metadata_item) for input_item, metadata_item in eval_data if metadata_item['qid'] in avail_qids]
    tasks = []
    with multiprocessing.Pool(processes=parallel_num) as pool:
        # 使用partial固定client参数
        func = partial(process_chunk_api, eval_func=eval_func, modality=modality, model_name=model_name, save_path=save_path)
        results = list(tqdm(
            pool.imap(func, eval_data),
            total=len(eval_data),
            desc=f'Processing API calls'
        ))
    
    return results

def prepare_prompt(prompt, task, segment_id, video_captions, subtitles):
    new_prompt = "Please answer the question based on the video and the text of audio content below.\n"
    aud_forms = task2audform[task]
    caption_text = merge_captions_segment_wise(video_captions, subtitles, aud_forms, segment_id=segment_id)
    new_prompt += caption_text +"\nHere's the question:\n"+ prompt
    return new_prompt
    
def evaluation_model(json_path, video_dir, modality, model_name, save_path, vid2timecode, caption_path='paired_captions.json', sub_path = 'subtitle'):

    # 加载JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    paired_captions = load_caption(caption_path)
    vid2sub = load_sub(sub_path)
    existing_results = []
    if os.path.exists(save_path):
        existing_results = load_jsonl(save_path)
    existing_qids = set(result['qid'] for result in existing_results)
    new_data = [result for result in data if result['qid'] not in existing_qids]

    # 处理每个条目
    inputs = list()
    metadata = list()
    for idx, item in enumerate(new_data):
        video_name = item.get('video_name')
        if not video_name:
            print(f"警告: 第 {idx} 条记录缺少 video_name，已跳过")
            continue
        qid = item.get('qid')
        task = item.get('task')
        segment_id = item.get('segment_id')  # 可能是索引(int)、索引列表(list)或None
        question = item.get('question')
        options = item.get('options')
        correct_answer = item.get('correct_answer')
        segments_list = vid2timecode[video_name]
        start_time, end_time = get_segment_time(segment_id, segments_list)
        # file_path = os.path.join(video_dir, video_name)
        file_path = os.path.join(video_dir, qid)
        
        prompt, correct_prefix, prefix2text = generate_mcq_prompt(question, options, correct_answer)
        if modality == 'v' or modality == 'av':
            # prompt = "Watch the video carefully and answer the question by directly outputting the correct option letter (e.g., 'A').\nQuestion: "+prompt
            prompt = "Watch the video carefully and answer the question with correct option letter (e.g., 'A').\nQuestion: "+prompt
        elif modality == 'a':
            prompt = "Listen to the audio carefully and answer the question by directly outputting the correct option letter (e.g., 'A').\nQuestion: "+prompt
            # prompt = "Listen to the audio carefully and answer the question with correct option letter (e.g., 'A').\nQuestion: "+prompt
        elif modality == 'vt':
            video_captions = paired_captions[video_name]
            if video_name in vid2sub:
                subtitles = clean_subtitles(vid2sub[video_name])
            else:
                subtitles = []
            prompt = prepare_prompt(prompt, task, segment_id, video_captions, subtitles)
        inputs.append((prompt, file_path, [start_time, end_time], prefix2text))
        # metadata.append([qid, video_name, segment_id, task, question, prompt, options, correct_answer, correct_prefix])
        metadata.append({
            'qid':qid,
            'video_name':video_name,
            'segment_id':segment_id,
            'duration':end_time-start_time,
            'task':task,
            'question':question,
            'question_prompt':prompt,
            'options':options,
            'correct_answer':correct_answer,
            'correct_prefix':correct_prefix,
        })
        
    if model_name == 'qwen2.5omni':
        from eval_qwenomni import evaluation
        results = eval_batch(inputs, metadata, evaluation, modality, model_name, save_path)
    elif model_name == 'omnir1':
        from eval_omnir1 import evaluation
        results = eval_multiprocess(inputs, metadata, evaluation, modality, model_name, save_path)
    elif model_name == 'videollama2':
        from eval_videollama2 import evaluation
        results = eval_multiprocess(inputs, metadata, evaluation, modality, model_name, save_path)
    elif model_name == 'videollama3':
        from eval_videollama3 import evaluation
        results = eval_multiprocess(inputs, metadata, evaluation, modality, model_name, save_path)
    elif model_name == 'vita1.5':
        from eval_vita import evaluation
        results = eval_multiprocess(inputs, metadata, evaluation, modality, model_name, save_path)
    elif model_name == 'qwen2.5vl':
        from eval_qwenvl import evaluation
        results = eval_batch(inputs, metadata, evaluation, modality, model_name, save_path)
    elif model_name == 'llavavideo':
        from eval_llavavideo import evaluation
        results = eval_multiprocess(inputs, metadata, evaluation, modality, model_name, save_path)
    elif model_name == "internvl":
        from eval_internvl import evaluation
        results = eval_multiprocess(inputs, metadata, evaluation, modality, model_name, save_path)
    elif model_name == 'kimiaudio':
        from eval_kimi import evaluation
        results = eval_multiprocess(inputs, metadata, evaluation, modality, model_name, save_path)
    elif model_name == 'avicuna':
        from eval_avicuna import evaluation
        results = eval_multiprocess(inputs, metadata, evaluation, modality, model_name, save_path)
    elif model_name == 'qwen2audio':
        from eval_qwenaudio import evaluation
        results = eval_batch(inputs, metadata, evaluation, modality, model_name, save_path)
    elif model_name == 'gemini':
        from eval_gemini import evaluation
        results = eval_api(inputs, metadata, evaluation, modality, model_name, save_path)
    elif model_name == 'onellm':
        model_results = load_json(f'./onellm_{modality}_result.json')
        results = list()
        qid2metadata = {}
        for item in metadata:
            qid2metadata[item['qid']] = item
        for item in model_results.values():
            meta = qid2metadata[item['qid']]
            meta['question_prompt'] = ''
            meta['correct_prefix'] = 'D'
            model_result = {
                    **meta,
                    'modality': modality,
                    'model': model_name,
                    'model_answer': item['answer'],
                    'model_output': '',
                }
            with open(save_path, 'a', encoding='utf-8') as f:
                json.dump(model_result, f, ensure_ascii=False)
                f.write('\n')
            results.append(model_result)
    elif model_name == 'gpt4o':
        model_results = load_json('./gpt4o_result.json')
        results = list()
        qid2metadata = {}
        for item in metadata:
            qid2metadata[item['qid']] = item
        for item in model_results:
            meta = qid2metadata[item['question_id']]
            meta['question_prompt'] = ''
            meta['correct_prefix'] = item['gt_answer']
            model_result = {
                    **meta,
                    'modality': 'v',
                    'model': model_name,
                    'model_answer': item['pred_letter'],
                    'model_output': item['answer'],
                }
            with open(save_path, 'a', encoding='utf-8') as f:
                json.dump(model_result, f, ensure_ascii=False)
                f.write('\n')
            results.append(model_result)
    elif model_name == 'salmonn':
        split_num = 2
        processed_data = list()
        results = list()
        for i in range(split_num):
            if os.path.exists(f'./salmonn_data/salmonn_{modality}_{i}_result.json'):
                processed_data += load_json(f'./salmonn_{modality}_{i}_result.json')
        if len(processed_data) >0:
            if os.path.exists(f'./salmonn_data/salmonn_{modality}_qid2metadata.json'):
                qid2metadata = load_json(f'./salmonn_{modality}_qid2metadata.json')
            for item in tqdm(processed_data):
                qid = item['qid']
                meta_data = qid2metadata[qid]
                text = item['gen_answer']
                model_answer = extract_model_answer(text, meta_data.pop('prefix2text'))
                model_result = {
                    **meta_data,
                    'modality': modality,
                    'model': model_name,
                    'model_answer': model_answer,
                    'model_output': text,
                }
                # Thread-safe file writing (optional: use file lock if needed)
                with open(save_path, 'a', encoding='utf-8') as f:
                    json.dump(model_result, f, ensure_ascii=False)
                    f.write('\n')
                results.append(model_result)
        else:
            # Shuffle data (while keeping inputs & metadata aligned)
            shuffled_inputs, shuffled_metadata = shuffle_data(inputs, metadata)
            qid2metadata = dict()
            
            # Split data into N chunks (N = number of GPUs)
            data_chunks = list(chunk_data(shuffled_inputs, shuffled_metadata, split_num))
            for i, (chunk_inputs, chunk_metadata) in enumerate(data_chunks):
                salmonn_inputs = list()
                for input_item, metadata_item in zip(chunk_inputs, chunk_metadata):
                    prompt, file_path, item_segment, prefix2text = input_item
                    metadata_item['prefix2text'] = prefix2text
                    qid2metadata[metadata_item['qid']] = metadata_item
                    if modality == 'av' or modality == 'a':
                        audio_path = file_path+'.wav'
                    elif modality == 'v':
                        audio_path = file_path + '_silent.wav'
                    salmonn_inputs.append({
                        "image_name":audio_path,
                        "conversation":[
                            {
                                "from": "human",
                                "value": prompt
                            },
                            {
                                "from": "gpt",
                                "value": "None"
                            }
                        ]
                    })
                save_json(salmonn_inputs, f'./salmonn_data/salmonn_{modality}_{i}.json')
            save_json(qid2metadata, f'./salmonn_data/salmonn_{modality}_qid2metadata.json')
    elif model_name == 'salmonno1':
        split_num = 2
        processed_data = list()
        results = list()
        for i in range(split_num):
            if os.path.exists(f'./salmonno1_data/salmonno1_{modality}_{i}_result.json'):
                processed_data += load_json(f'./salmonno1_data/salmonno1_{modality}_{i}_result.json')
        if len(processed_data) >0:
            if os.path.exists(f'./salmonno1_data/salmonno1_{modality}_qid2metadata.json'):
                qid2metadata = load_json(f'./salmonno1_data/salmonno1_{modality}_qid2metadata.json')
            for item in tqdm(processed_data):
                qid = item['qid'][0]
                meta_data = qid2metadata[qid]
                text = item['answer']
                model_answer = extract_model_answer(text, meta_data.pop('prefix2text'))
                model_result = {
                    **meta_data,
                    'modality': modality,
                    'model': model_name,
                    'model_answer': model_answer,
                    'model_output': text,
                }
                # Thread-safe file writing (optional: use file lock if needed)
                with open(save_path, 'a', encoding='utf-8') as f:
                    json.dump(model_result, f, ensure_ascii=False)
                    f.write('\n')
                results.append(model_result)
        else:
            os.makedirs('./salmonno1_data', exist_ok=True)
            # Shuffle data (while keeping inputs & metadata aligned)
            shuffled_inputs, shuffled_metadata = shuffle_data(inputs, metadata)
            qid2metadata = dict()
            
            # Split data into N chunks (N = number of GPUs)
            data_chunks = list(chunk_data(shuffled_inputs, shuffled_metadata, split_num))
            for i, (chunk_inputs, chunk_metadata) in enumerate(data_chunks):
                salmonn_inputs = list()
                for input_item, metadata_item in zip(chunk_inputs, chunk_metadata):
                    prompt, file_path, item_segment, prefix2text = input_item
                    metadata_item['prefix2text'] = prefix2text
                    qid2metadata[metadata_item['qid']] = metadata_item
                    multimodal_path = {}
                    if 'v' in modality:
                        multimodal_path["video"] = file_path+'.mp4'
                    if 'a' in modality:
                        multimodal_path["audio"] = file_path+'.wav'
                        
                    salmonn_inputs.append({
                        "qid":metadata_item['qid'],
                        **multimodal_path,
                        "conversations":[
                            {
                                "from": "human",
                                "value": prompt
                            },
                            {
                                "from": "gpt",
                                "value": "None"
                            }
                        ]
                    })
                save_json(salmonn_inputs, f'./salmonno1_data/salmonno1_{modality}_{i}.json')
            save_json(qid2metadata, f'./salmonno1_data/salmonno1_{modality}_qid2metadata.json')            
    elif model_name == 'gemini_api':
        processed_data = list()
        results = list()
        if os.path.exists(f'./gemini_{modality}_result.json'):
            processed_data = load_json(f'./gemini_{modality}_result.json')
        if len(processed_data) >0:
            if os.path.exists(f'./gemini_{modality}_qid2metadata.json'):
                qid2metadata = load_json(f'./gemini_{modality}_qid2metadata.json')
            for item in tqdm(processed_data):
                qid = item['qid']
                meta_data = qid2metadata[qid]
                text = item['model_output']
                model_answer = extract_model_answer(text, meta_data.pop('prefix2text'))
                model_result = {
                    **meta_data,
                    'modality': modality,
                    'model': model_name,
                    'model_answer': model_answer,
                    'model_output': text,
                }
                # Thread-safe file writing (optional: use file lock if needed)
                with open(save_path, 'a', encoding='utf-8') as f:
                    json.dump(model_result, f, ensure_ascii=False)
                    f.write('\n')
                results.append(model_result)
        else:
            assert 1==0
            qid2metadata = dict()
            gemini_inputs = list()
            upload_files = list()
            for input_item, metadata_item in zip(inputs, metadata):
                prompt, file_path, item_segment, prefix2text = input_item
                video_path = os.path.basename(file_path+'.mp4')
                metadata_item['prefix2text'] = prefix2text
                qid2metadata[metadata_item['qid']] = metadata_item
                gemini_inputs.append({
                    'qid':metadata_item['qid'],
                    "prompt": prompt,
                    "video_path":video_path,
                })
                upload_files.append(video_path)
            save_json(gemini_inputs, f'./gemini_{modality}.json')
            save_json(qid2metadata, f'./gemini_{modality}_qid2metadata.json')
            save_json(upload_files, f'./gemini_upload_files.json')
    elif model_name == 'aurelia':
        model_results = load_json('./aurelia_result_av.json')
        results = list()
        qid2metadata = {}
        for item in metadata:
            qid2metadata[item['qid']] = item
        for qid in model_results:
            item = model_results[qid]
            meta = qid2metadata[qid]
            meta['question_prompt'] = item['query']
            meta['correct_prefix'] = item['ground_truth']
            model_result = {
                    **meta,
                    'modality': 'av',
                    'model': model_name,
                    'model_answer': item['final_answer'],
                    'model_output': item['final_answer'],
                }
            with open(save_path, 'a', encoding='utf-8') as f:
                json.dump(model_result, f, ensure_ascii=False)
                f.write('\n')
            results.append(model_result)
        
    return results

def main(args):
    qa_path = args.qa_path
    # video_dir = '/data-01/jianghan/benchmark'
    video_dir = '/data-01/jianghan/qa_vid'
    modality = args.modality
    model_name = args.model_name
    save_path = f'./results/eval_results_{model_name}_{modality}.jsonl'
    vid2timecode = load_json('./vid2timecode.json')
    # 处理视频
    results = evaluation_model(
        json_path=qa_path,
        video_dir=video_dir,
        modality = modality, 
        model_name = model_name,
        save_path = save_path,
        vid2timecode=vid2timecode,
    )
    with open('./results.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)   
    
if __name__ == "__main__":
    model_list = ["qwen2.5omni", "videollama2", "videollama3", "vita1.5","salmonn", "qwen2.5vl", "llavavideo", 
                  "internvl", "kimiaudio", "onellm", "qwen2audio", "gpt4o", "gemini", "gemini_api", "omnir1", 
                  "salmonno1", "avicuna", "aurelia"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', choices=["a", "v", "av", "vt"], help='', required=False, default = 'av')
    parser.add_argument('--qa-path', help='', required=True)
    parser.add_argument('--model-name', choices=model_list, help='', required=True)
    args = parser.parse_args()
    main(args)
