import os
# Set retry limit for video decoding
os.environ['DECORD_EOF_RETRY_MAX'] = '40960'

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
    Extract the answer choice from model output using multiple matching strategies:
    1. Regex pattern matching for common answer formats
    2. Direct prefix searching (e.g., "A.", "B.")
    3. Option text content matching
    4. Lenient matching for very short outputs

    Args:
        model_output: Raw text output from the model
        prefix2text: Mapping from option prefixes to their text content, e.g., {'A': 'Paris', 'B': 'London'}
        possible_prefixes: List of valid option prefixes, defaults to ['A', 'B', 'C', 'D']
        answer_patterns: List of regex patterns for answer extraction

    Returns:
        Single matched option prefix (e.g., 'A'), or None if multiple/no matches found
    """
    # Set default parameters
    if possible_prefixes is None:
        possible_prefixes = ['A', 'B', 'C', 'D']
    
    if answer_patterns is None:
        answer_patterns = [
            r'answer(?: is)?\s*[:=-]?\s*(["\'(]?\s*[A-D]\s*["\')]?)',  # "Answer: A", "answer is 'B'", "answer is (C)"
            r'([A-D])(?:\s*is the|\)?\s*is)\s*(?:correct|right|answer)',  # "A is correct", "B) is right", "C is the answer"
            r'option\s*([A-D])\b',  # "option A", "choose option C"
            r'select(?:ed)?\s*([A-D])\b',  # "select B", "selected D"
            r'["\'(]\s*([A-D])\s*["\')]\s*(?:is|as)\s*(?:the )?(?:answer|correct)',  # "'A' is the answer", "(B) is correct"
        ]

    # Preprocessing: remove newlines and extra whitespace
    cleaned_text = ' '.join(model_output.split())

    # Strategy 1: Try all regex matching patterns
    found_prefixes = set()
    for pattern in answer_patterns:
        matches = re.finditer(pattern, cleaned_text, re.IGNORECASE)
        for match in matches:
            extracted = match.group(1).upper()
            if extracted in possible_prefixes:
                found_prefixes.add(extracted)

    # Strategy 2: Direct prefix search (e.g., "A.")
    for prefix in possible_prefixes:
        if re.search(rf'\b{prefix}\b(?=\.\s)', cleaned_text):
            found_prefixes.add(prefix.upper())

    # Strategy 3: Check if option text content is in output
    for prefix, text in prefix2text.items():
        if prefix.upper() in possible_prefixes:
            if text in cleaned_text or text.lower() in cleaned_text:
                found_prefixes.add(prefix.upper())

    # Strategy 4: Lenient matching for very short outputs (single word)
    if len(cleaned_text.split()) == 1:
        for prefix in possible_prefixes:
            if prefix in cleaned_text.strip().upper():
                found_prefixes.add(prefix)

    # Return None if multiple matches found (ambiguous)
    if len(found_prefixes) > 1:
        return None

    # Return the single matched prefix
    if len(found_prefixes) == 1:
        return found_prefixes.pop()
    return None

def generate_mcq_prompt(
    question: str,
    options: List[str],
    correct_answer: str,
    option_prefixes: List[str] = None
) -> Tuple[str, str, Dict[str, str]]:
    """
    Generate a multiple-choice question prompt with shuffled options.

    This function creates a formatted MCQ prompt by:
    1. Shuffling the options to prevent position bias
    2. Assigning prefixes (A, B, C, D) to shuffled options
    3. Tracking which prefix corresponds to the correct answer

    Args:
        question: The question text
        options: List of option texts
        correct_answer: The correct answer text (must be in options list)
        option_prefixes: Custom option prefixes, defaults to ['A', 'B', 'C', 'D']

    Returns:
        tuple: (formatted_prompt, correct_prefix, prefix_to_text_mapping)
            - formatted_prompt: Question with labeled options
            - correct_prefix: The prefix (e.g., 'A') of the correct answer
            - prefix_to_text_mapping: Dict mapping each prefix to its option text

    Raises:
        ValueError: If fewer than 2 options, too many options, or correct_answer not in options
    """
    prefix2text = {}

    # Set default option prefixes (A, B, C, D)
    if option_prefixes is None:
        option_prefixes = ['A', 'B', 'C', 'D']

    # Validate inputs
    if len(options) < 2:
        raise ValueError("Must provide at least 2 options")
    if len(options) > len(option_prefixes):
        raise ValueError(f"Number of options ({len(options)}) exceeds number of prefixes ({len(option_prefixes)})")
    if correct_answer not in options:
        raise ValueError("Correct answer not in provided options")

    # Shuffle options (but track original indices)
    indexed_options = list(enumerate(options))
    random.shuffle(indexed_options)

    # Build prefixed option text and track correct answer
    prefixed_options = []
    correct_prefix = None

    for idx, (original_idx, option) in enumerate(indexed_options):
        prefix = option_prefixes[idx]
        prefixed_option = f"{prefix}. {option}"
        prefixed_options.append(prefixed_option)
        prefix2text[prefix] = option

        # Check if this is the correct answer
        if options[original_idx] == correct_answer:
            correct_prefix = prefix

    # Combine question and options
    options_text = "\n".join(prefixed_options)
    prompt = f"{question}\n{options_text}"

    return prompt, correct_prefix, prefix2text

def get_segment_time(segment_id, segments_list):
    """
    Calculate start and end times for video segments.

    Args:
        segment_id: Can be:
            - int: Single segment index
            - list: Multiple segment indices (uses min/max range)
            - None: Entire video (all segments)
        segments_list: List of video segments, where each segment is [start_time, end_time]

    Returns:
        List[float]: [start_time, end_time] for the requested segment(s)
    """
    if isinstance(segment_id, int):
        start_segment, end_segment = segment_id, segment_id
    elif isinstance(segment_id, list):
        start_segment, end_segment = min(segment_id), max(segment_id)
    else:  # None - use entire video
        start_segment, end_segment = 0, -1
    start_time = segments_list[start_segment][0]
    end_time = segments_list[end_segment][-1]
    return [start_time, end_time]

def batch_inputs_num(inputs, metadata, max_num=5):
    """
    Batch inputs and metadata into fixed-size chunks for processing.

    This generator yields batches of up to max_num items, maintaining
    alignment between inputs and metadata.

    Args:
        inputs: List of input data items
        metadata: List of metadata items (must align with inputs)
        max_num: Maximum number of items per batch (default: 5)

    Yields:
        tuple: (batch_inputs, batch_metadata) for each batch
    """
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
    """
    Evaluate model on batched inputs and save results incrementally.

    This function:
    1. Batches inputs using batch_inputs_num()
    2. Calls the evaluation function for each batch
    3. Extracts answers from model outputs
    4. Saves results line-by-line to JSONL file

    Args:
        inputs: List of (prompt, file_path, time_range, prefix2text) tuples
        metadata: List of metadata dictionaries for each input
        evaluation: Evaluation function that processes batches
        modality: Evaluation modality ('a', 'v', 'av', 'vt')
        model_name: Name of the model being evaluated
        save_path: Path to save results (JSONL format)

    Returns:
        List of result dictionaries containing model answers and metadata
    """
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
    """
    Split inputs and metadata into N approximately equal chunks.

    Uses ceiling division to ensure all items are included. The last chunk
    may be smaller if items don't divide evenly.

    Args:
        inputs: List of input items to split
        metadata: List of metadata items (must align with inputs)
        num_chunks: Number of chunks to create

    Yields:
        tuple: (chunk_inputs, chunk_metadata) for each chunk
    """
    chunk_size = (len(inputs) + num_chunks - 1) // num_chunks  # Round up division
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        yield inputs[start:end], metadata[start:end]

def process_chunk(task, eval_func, modality, model_name, save_path):
    """
    Process a data chunk on an assigned GPU device (multiprocessing worker).

    This function:
    1. Loads the model on the specified GPU device
    2. Processes each item in the chunk sequentially
    3. Extracts answers from model outputs
    4. Thread-safely writes results to file using fcntl locks

    Args:
        task: Tuple of (chunk_inputs, chunk_metadata, device)
        eval_func: Model-specific evaluation function
        modality: Evaluation modality ('a', 'v', 'av', 'vt')
        model_name: Name of the model to load
        save_path: Output file path for results

    Returns:
        List of result dictionaries for this chunk

    Note:
        Uses fcntl.flock for thread-safe file writing across processes
    """
    chunk_inputs, chunk_metadata, device = task

    # Add random delay to stagger GPU initialization
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
    """
    Evaluate model using multiprocessing across available GPUs.

    Workflow:
    1. Detects available GPUs (or falls back to CPU)
    2. Shuffles data while maintaining input-metadata alignment
    3. Splits data into chunks (one per GPU)
    4. Assigns each chunk to a GPU device
    5. Processes chunks in parallel using multiprocessing.Pool

    Args:
        inputs: List of input data items
        metadata: List of metadata items (aligned with inputs)
        eval_func: Model-specific evaluation function
        modality: Evaluation modality ('a', 'v', 'av', 'vt')
        model_name: Name of the model being evaluated
        save_path: Output file path for results

    Returns:
        List of lists of result dictionaries (one list per chunk)

    Note:
        Uses 'spawn' start method for compatibility with CUDA
    """
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
    """
    Dynamically load a model on the specified device.

    This function imports and loads the appropriate model based on model_name.
    Each model has its own eval_<model>.py module with a load_model function.

    Args:
        model_name: Name of the model to load (e.g., 'videollama2', 'omnir1')
        device: Device string (e.g., 'cuda:0', 'cpu')

    Returns:
        Model configuration object specific to the loaded model

    Raises:
        ValueError: If model_name is not recognized

    Supported models:
        videollama2, omnir1, videollama3, vita1.5, llavavideo,
        internvl, kimiaudio, avicuna
    """
    # Model name to module name mapping
    MODEL_LOADERS = {
        'videollama2': 'eval_videollama2',
        'omnir1': 'eval_omnir1',
        'videollama3': 'eval_videollama3',
        'vita1.5': 'eval_vita',
        'llavavideo': 'eval_llavavideo',
        'internvl': 'eval_internvl',
        'kimiaudio': 'eval_kimi',
        'avicuna': 'eval_avicuna',
    }

    if model_name not in MODEL_LOADERS:
        raise ValueError(f"Invalid model name: {model_name}")

    module_name = MODEL_LOADERS[model_name]
    module = __import__(module_name, fromlist=['load_model'])
    return module.load_model(device)

def process_chunk_api(task, eval_func, modality, model_name, save_path):
    """
    Process a chunk of data using API-based evaluation with rate limiting.

    Similar to process_chunk but designed for API-based models which don't
    require GPU assignment. Includes random delays to avoid rate limiting.

    Args:
        task: Tuple of (chunk_inputs, chunk_metadata, _unused)
        eval_func: API evaluation function
        modality: Evaluation modality
        model_name: Name of the API model
        save_path: Output file path

    Returns:
        List of result dictionaries for this chunk
    """
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
    """
    Evaluate using API-based models with parallel requests.

    This function:
    1. Filters inputs based on available question IDs
    2. Processes requests in parallel using multiprocessing
    3. Includes random delays to avoid rate limiting

    Args:
        inputs: List of input data items
        metadata: List of metadata items
        eval_func: API evaluation function
        modality: Evaluation modality
        model_name: Name of the API model
        save_path: Output file path

    Returns:
        List of result dictionaries (may include None for failed requests)

    Note:
        Parallel level is hardcoded to 3 processes
    """
    avail_qids = load_json('./avail_qids.json')
    parallel_num = 3
    eval_data = list(zip(inputs, metadata))
    eval_data = [(input_item, metadata_item) for input_item, metadata_item in eval_data if metadata_item['qid'] in avail_qids]

    with multiprocessing.Pool(processes=parallel_num) as pool:
        # Use partial to fix common parameters
        func = partial(process_chunk_api, eval_func=eval_func, modality=modality, model_name=model_name, save_path=save_path)
        results = list(tqdm(
            pool.imap(func, eval_data),
            total=len(eval_data),
            desc=f'Processing API calls'
        ))

    return results

def prepare_prompt(prompt, task, segment_id, video_captions, subtitles):
    """
    Prepare prompt for text-based (VT) modality evaluation.

    Combines video captions and subtitles with the question to create
    a text-only prompt that contains audio-visual content information.

    Args:
        prompt: Original question prompt
        task: Task type (used to determine audio format)
        segment_id: Video segment identifier
        video_captions: List of video caption segments
        subtitles: List of subtitle segments

    Returns:
        str: Enhanced prompt with caption and subtitle context
    """
    new_prompt = "Please answer the question based on the video and the text of audio content below.\n"
    aud_forms = task2audform[task]
    caption_text = merge_captions_segment_wise(video_captions, subtitles, aud_forms, segment_id=segment_id)
    new_prompt += caption_text + "\nHere's the question:\n" + prompt
    return new_prompt
    
def evaluation_model(json_path, video_dir, modality, model_name, save_path, vid2timecode, caption_path='paired_captions.json', sub_path='subtitle'):
    """
    Main evaluation pipeline for processing questions and running model inference.

    This function:
    1. Loads questions from JSON and filters out already-processed items
    2. Prepares inputs (prompts, file paths, time segments) for each question
    3. Routes to appropriate evaluation function based on model_name
    4. Handles model-specific data formatting and result processing

    Args:
        json_path: Path to questions JSON file
        video_dir: Directory containing video files
        modality: Evaluation modality - 'a' (audio), 'v' (video), 'av' (audio-video), 'vt' (video-text)
        model_name: Name of model to evaluate (determines routing)
        save_path: Output file path for results (JSONL format)
        vid2timecode: Mapping from video names to segment timecodes
        caption_path: Path to video captions JSON (for VT modality)
        sub_path: Path to subtitle files directory (for VT modality)

    Returns:
        List of result dictionaries containing model answers and metadata

    Model routing:
        - Batch processing: qwen2.5omni, qwen2.5vl, qwen2audio
        - Multiprocessing: videollama2, omnir1, videollama3, vita1.5, llavavideo,
                          internvl, kimiaudio, avicuna
        - API: gemini
        - Special handling: onellm, gpt4o, salmonn, salmonno1, gemini_api, aurelia

    Note:
        - Skips items with missing video_name
        - Maintains alignment between inputs and metadata
        - Some models require pre-generated results (onellm, gpt4o, aurelia)
    """
    # Load questions from JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    paired_captions = load_caption(caption_path)
    vid2sub = load_sub(sub_path)

    # Load existing results to avoid reprocessing
    existing_results = []
    if os.path.exists(save_path):
        existing_results = load_jsonl(save_path)
    existing_qids = set(result['qid'] for result in existing_results)
    new_data = [result for result in data if result['qid'] not in existing_qids]

    # Process each question and prepare inputs
    inputs = list()
    metadata = list()
    for idx, item in enumerate(new_data):
        video_name = item.get('video_name')
        if not video_name:
            print(f"Warning: Record {idx} missing video_name, skipping")
            continue
        qid = item.get('qid')
        task = item.get('task')
        segment_id = item.get('segment_id')  # Can be int (single segment), list (multiple), or None (full video)
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
    # Use video_dir from args or environment variable
    video_dir = args.video_dir if hasattr(args, 'video_dir') and args.video_dir else os.environ.get('VIDEO_DIR', './data/videos')
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
                  "internvl", "kimiaudio", "onellm", "qwen2audio", "gpt4o", "gemini", "omnir1", 
                  "salmonno1", "avicuna", "aurelia"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', choices=["a", "v", "av", "vt"], help='', required=False, default = 'av')
    parser.add_argument('--qa-path', help='', required=True)
    parser.add_argument('--video-dir', help='Directory containing video files', required=False, default='./data/videos')
    parser.add_argument('--model-name', choices=model_list, help='', required=True)
    args = parser.parse_args()
    main(args)
