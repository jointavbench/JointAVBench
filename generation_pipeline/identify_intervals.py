"""
Interval Identification Module

This script identifies video segment intervals that are relevant to generated Q&A pairs.
It uses LLM (via OpenAI API) to analyze captions and determine which video segments
contain the information needed to answer each question.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
import json
from collections import defaultdict
from tqdm import tqdm
import time
import random
from openai import OpenAI
from utils import load_caption, load_json, save_json, load_sub, join_subtitles, merge_captions_segment_wise, join_captions
from prompts import INTERVAL_CHECK_PROMPT, STAGE2TASK
from parse_data import clean_subtitles, extract_segment_interval

# Configuration
MODEL_PATH = os.environ.get('QWEN_MODEL_PATH', 'Qwen/Qwen2.5-72B-Instruct')
STAGE = 'multi'  # Can be 'single', 'multi', or 'full'
API_KEY = os.environ.get('QWEN_API_KEY', '')
MODEL_NAME = "qwen2.5-72b-instruct"

SYSTEM_PROMPT = """
Your task is to identify the segment intervals based on the information used in the question-answer pair. Please follow the instructions to identify the interval.
"""

# Initialize OpenAI client with Qwen API endpoint
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def process_openai_api(messages, max_retries=5):
    """
    Process a single API call to identify segment intervals with retry mechanism.

    This function makes API calls to the LLM to identify which video segments
    are relevant for answering a question. It includes exponential backoff
    retry logic to handle transient failures.

    Args:
        messages: Tuple of (message_list, metadata) where:
            - message_list: List of chat messages for the API
            - metadata: List containing video_name and other info
        max_retries: Maximum number of retry attempts (default: 5)

    Returns:
        tuple: (metadata, generated_text) where:
            - metadata: Original metadata passed in
            - generated_text: LLM response with segment interval, or None if failed

    Note:
        Uses exponential backoff with random jitter for retries
    """
    message, metadata = messages
    video_name = metadata[0]

    # Add random delay to avoid rate limiting
    time.sleep(random.randint(1, 3))

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
                # Exponential backoff: 2, 4, 8, 16 seconds + random jitter (max 30s)
                wait_time = min(2 ** retry_count + random.uniform(0, 1), 30)
                print(f"Video {video_name} attempt {retry_count} failed: {str(e)}, waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"Video {video_name} reached max retries ({max_retries}), error: {str(e)}")
                return metadata, None

if __name__ == '__main__':
    """
    Main execution block for interval identification.

    Workflow:
    1. Load video captions and Q&A pairs
    2. For each task type in the specified STAGE:
       - Filter videos with required audio content
       - Generate interval prompts from captions + Q&A
       - Call LLM to identify relevant segment intervals
       - Save identified intervals to JSON files

    Output:
        Creates {task}_intervals.json files containing segment boundaries
        for each question-answer pair.
    """
    # Configuration paths
    caption_path = 'paired_captions.json'
    sub_path = './raw_data/subtitle'
    save_path = './results'

    # Load data
    paired_captions = load_caption(caption_path)
    vid2sub = load_sub(sub_path)
    vid2covid = load_json('./vid2covid.json')
    vid_names = list(paired_captions.keys())

    # Load Q&A pairs organized by task type
    qa_path = './task2qa_general.json'
    task2qa = load_json(qa_path)

    # Process tasks for the specified stage (single/multi/full)
    scene_level = STAGE2TASK[STAGE]
    for task in scene_level:
        # Filter tasks to process (can be configured)
        if task not in ['task9', 'task13']:
            continue
        qas = list()
        aud_forms = scene_level[task]  # Required audio types for this task
        generated_qas = task2qa[task]
        vid2qa = defaultdict(list)

        # Load existing results if available (resume capability)
        if os.path.exists(os.path.join(save_path, f'{task}_intervals.json')):
            generated_qas = load_json(os.path.join(save_path, f'{task}_intervals.json'))
            for item in generated_qas:
                if item['segment_id'] is not None:
                    # Already has interval identified, keep it
                    qas.append(item)
                    continue
                # Need to identify interval for this item
                video_name = item['video_name']
                vid2qa[video_name].append(item)
        else:
            # First run, process all Q&A pairs
            for item in generated_qas:
                video_name = item['video_name']
                vid2qa[video_name].append(item)
        vid_names = list(vid2qa.keys())
        vid_names.sort()

        for vid in tqdm(vid_names, desc=f'Processing [{task}]'):
            # Get valid segments and captions for this video
            covid = vid2covid[vid]
            video_captions = paired_captions[vid]
            video_captions = [i for i in video_captions if i['segment_id'] in covid]

            # Load and process subtitles and audio captions
            subtitles = clean_subtitles(vid2sub[vid])
            joined_sub = join_subtitles(subtitles)
            vid_caption, music_caption, sound_caption, speech_emotion = join_captions(video_captions)

            # Validate that video has required audio content for this task
            if 'speech' in aud_forms and joined_sub == '':
                continue
            if 'speech_emotion' in aud_forms and speech_emotion == '':
                continue
            if 'sound_event' in aud_forms and sound_caption == '':
                continue
            if 'music' in aud_forms and music_caption == '':
                continue
            # Prepare segment information for prompting
            prompt_completion = dict()
            segments_info = merge_captions_segment_wise(video_captions, subtitles, aud_forms)
            prompt_completion['segments_info'] = segments_info

            # Process each Q&A pair for this video
            available_qas = vid2qa[vid]
            for available_qa in available_qas:
                prompt_completion['question'] = available_qa['question']
                prompt_completion['answer'] = available_qa['answer']
                prompt_completion['explanation'] = available_qa['explanation']

                # Generate prompt and call LLM to identify intervals
                task_prompt = INTERVAL_CHECK_PROMPT.format(**prompt_completion)
                message = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task_prompt}
                ]
                _, generated_text = process_openai_api([message, [vid]])

                # Parse interval from LLM response
                segment_info = extract_segment_interval(generated_text)
                if segment_info is None:
                    intervals = None
                else:
                    intervals = [segment_info['start'], segment_info['end']]

                # Store result
                caption = {
                    'qid': available_qa['qid'],
                    'video_name': vid,
                    'task': task,
                    'segment_id': intervals,
                    'question': available_qa['question'],
                    'answer': available_qa['answer'],
                    'explanation': available_qa['explanation'],
                    'caption': generated_text,
                }
                qas.append(caption)

        # Save results for this task
        save_json(qas, os.path.join(save_path, f'{task}_intervals.json'))

        