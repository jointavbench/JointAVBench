import os
from utils import load_json, save_json, seconds2timestamp
from tqdm import tqdm
import pandas as pd
import re
import pickle
import random
from prompts import STAGE2TASK
from collections import defaultdict


def extract_se_output(output):
    # output_text = output.pop('caption')
    output_text = output['caption']
    output_pattern = r"\[Output\](.*)"
    match = re.search(output_pattern, output_text, re.DOTALL)
    text = ''
    if match:
        output_content = match.group(1).strip()
        
        if "[Unavailable]" in output_content:
            text = ""  
        else:
            text = output_content  # 可行，返回提取的文本
    else:
        text = ""  
    return {
        **output,
        'generated_output':output_text,
        'speech_emotion':text,
    }
    
def extract_ac_output(output):
    output_text = output.pop('caption')
    # output_text = output['caption']
    # Define regex patterns to extract [music] and [sound event] sections
    music_pattern = r"\[music\](.*?)(?=\[sound event\]|$)"
    sound_event_pattern = r"\[sound event\](.*?)(?=\[music\]|$)"

    # Extract [music] and [sound event] sections
    music_match = re.search(music_pattern, output_text, re.DOTALL)
    sound_event_match = re.search(sound_event_pattern, output_text, re.DOTALL)

    # Validate the extracted sections
    if music_match:
        music = music_match.group(1).strip()
        if music.lower() == 'none':
            music = ''
    else:
        music = ''
    if sound_event_match:
        sound_event = sound_event_match.group(1).strip()
        if sound_event.lower() == 'none':
            sound_event = ''
    else:
        sound_event = ''
    return {
        **output,
        'generated_output':output_text,
        "music": music,
        "sound_event": sound_event,
    }

def output_processing(task, output):
    """
    Processes the output of a specific task based on its type.
    Returns the extracted content or None if the output is invalid.
    """
    if task == "audio_caption_cleaning":
        return extract_ac_output(output)
    elif task == "speech_emotion_cleaning":
        return extract_se_output(output)
    else:
        raise ValueError(f"Unknown task: {task}")

def extract_segment_interval(text):
    """
    Extracts start segment, end segment, and rationale from the given text format.
    
    Args:
        text (str): Input text in the specified format
        
    Returns:
        dict: Dictionary containing 'start', 'end', and 'rationale' keys
        None: If the pattern doesn't match
        
    Example:
        input_text = '''[Start]: 15
        [End]: 18
        [Rationale]: Character motivations explained in 15, key action in 16-17, resolution in 18'''
        
        extract_segment_info(input_text)
        # Returns: {'start': 15, 'end': 18, 'rationale': 'Character motivations explained...'}
    """
    pattern = r'\[Start\]:\s*(\d+).*?\[End\]:\s*(\d+).*?\[Rationale\]:\s*(.+)'
    
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return {
            'start': int(match.group(1)),
            'end': int(match.group(2)),
            'rationale': match.group(3).strip()
        }
    return None

def clean_subtitles(subtitles, max_repeats=2):
    if not subtitles:
        return subtitles

    cleaned_subtitles = []
    i = 0
    n = len(subtitles)
    
    while i < n:
        current_text = subtitles[i]['text']
        j = i + 1
        
        # 找出连续重复的句子
        while j < n and subtitles[j]['text'] == current_text:
            j += 1
            
        repeat_count = j - i
        
        # 如果重复次数不超过最大允许次数，保留这些句子
        if repeat_count <= max_repeats:
            cleaned_subtitles.extend(subtitles[i:j])
        # 否则，跳过所有重复的句子
        
        i = j  # 移动到下一组不同的句子
    
    # 保留的ASCII字符过滤（原功能保持不变）
    filtered_subtitles = []
    for subtitle in cleaned_subtitles:
        text = subtitle['text']
        if all(ord(char) < 128 for char in text):
            filtered_subtitles.append(subtitle)
            
    return filtered_subtitles

def parse_qa_explanation(text):
    # Split the text by the given sections
    sections = text.split('\n')
    
    # Initialize empty dictionary to hold the parsed data
    parsed_data = {
        'question': '',
        'answer': '',
        'explanation': ''
    }
    
    # Loop through each section and extract the relevant parts
    for section in sections:
        if section.startswith('[Question]'):
            parsed_data['question'] = section[len('[Question] '):].strip()
        elif section.startswith('[Answer]'):
            parsed_data['answer'] = section[len('[Answer] '):].strip()
        elif section.startswith('[Explanation]'):
            parsed_data['explanation'] = section[len('[Explanation] '):].strip()
    
    return parsed_data

def json_to_excel(json_data, excel_file):
    if type(json_data) == str:
        data = json.loads(json_data)
    else:
        data = json_data
    df = pd.DataFrame(data)    
    df.to_excel(excel_file, index=False)

def parse_single_scene(qa_data_path, save_excel = ''):
    qa_data = load_json(qa_data_path)
    results = list()
    for qa in qa_data:
        if '[Unavailable]' in qa['caption']:
            continue
        else:
            parsed_data = parse_qa_explanation(qa['caption'])
            results.append({
                'task':qa['task'],
                'video_name':qa['video_name'],
                "segment_id":qa['segment_id'],
                'start_time':qa['start_time'],
                'end_time':qa['end_time'],
                'question':parsed_data['question'],
                'answer':parsed_data['answer'],
                'explanation':parsed_data['explanation'],
            })
    if save_excel:
        json_to_excel(results, save_excel)
    return results

def extract_outputs(large_string):
    # 使用正则表达式提取每个 Output 块
    output_pattern = r"<Output \d+>\n(.*?)(?=<Output \d+>|$)"  # 匹配每个 Output 的内容
    outputs = re.findall(output_pattern, large_string, re.DOTALL)  # 提取所有 Output 块
    return outputs

def extract_details(output):
    # 正则表达式匹配每一部分的信息
    question_pattern = r"\[Question\] (.*?)\n"
    answer_pattern = r"\[Answer\] (.*?)\n"
    explanation_pattern = r"\[Explanation\] (.*?)\n"
    
    # 使用正则表达式提取信息
    question = re.search(question_pattern, output)
    answer = re.search(answer_pattern, output)
    explanation = re.search(explanation_pattern, output)
    
    # 构造结果字典
    result = {
        "question": question.group(1) if question else '',
        "answer": answer.group(1) if answer else '',
        "explanation": explanation.group(1) if explanation else '',
    }
    
    return result

def parse_multi_scene(task_path, save_excel = ''):
    task_outputs = load_json(task_path)
    parsed_outputs = list()
    seen_questions = set()
    for task_output in task_outputs:
        output = task_output.pop('caption')
        vid_name = task_output['video_name']
        output_qas = extract_outputs(output)
        for qa in output_qas:
            result = extract_details(qa)
            question = result.get('question', '')
            if (vid_name,question) not in seen_questions:
                seen_questions.add((vid_name, question))       
                parsed_outputs.append({
                    **task_output,
                    **result,
                })
    if save_excel:
        json_to_excel(parsed_outputs, save_excel)
    return parsed_outputs

def parse_judgements(file_path):
    task_judgements = load_json(file_path)
    for judgement in task_judgements:
        if judgement['judgement_text'] is None:
            judgement['judgement'] = 'NO'
        else:
            if '[YES]' in judgement['judgement_text']:
                judgement['judgement'] = 'YES'
            elif '[NO]' in judgement['judgement_text']:
                judgement['judgement'] = 'NO'
            else:
                judgement['judgement'] = 'UNKNOWN'
    return task_judgements

def gather_judgements(save_dir, stage, quality_judge = False, save_excel = ''):
    tasks = STAGE2TASK[stage]
    original_list = list()
    judgement_list = list()
    merged_list = list()
    if stage == 'single':
        for task in tasks:
            task_judgements = parse_judgements(os.path.join(save_dir, task+'_modality_judgements.json'))
            judgement_list += task_judgements
            parsed_qa = parse_single_scene(os.path.join(save_dir, task+'.json'))
            original_list += parsed_qa
        df1 = pd.DataFrame(original_list)
        df2 = pd.DataFrame(judgement_list)
        merged_df = pd.merge(df1, df2, on=['video_name', 'segment_id', 'type'], how='inner')
    else:
        for task in tasks:
            parsed_qa = parse_multi_scene(os.path.join(save_dir, task+'.json'))
            original_list += parsed_qa
            task_judgements = parse_judgements(os.path.join(save_dir, task+'_simple_judgements.json'))
            judgement_list += task_judgements
        df1 = pd.DataFrame(original_list)
        df2 = pd.DataFrame(judgement_list)
        merged_df = pd.merge(df1, df2, on=['video_name', 'type', 'question'], how='inner')
    if quality_judge:
        quality_list = list()
        for task in tasks:
            task_judgements = parse_judgements(os.path.join(save_dir, task+'_judgements.json'))
            quality_list += task_judgements
        quality_df = pd.DataFrame(quality_list)
        if stage == 'single':
            merged_df = pd.merge(merged_df, quality_df, on=['video_name', 'segment_id', 'type'], how='inner')
        else:
            merged_df = pd.merge(merged_df, quality_df, on=['video_name', 'type', 'question'], how='inner')
    merged_list = merged_df.to_dict(orient='records')

    if save_excel:
        merged_df.to_excel(save_excel, index=False)
    return merged_list

def update_segment_interval(save_dir):
    task2qa = defaultdict(list)
    for task in ['task9', 'task10', 'task12']:
        task_interval = load_json(os.path.join(save_dir, task+'_intervals.json'))
        for task_qa in task_interval:
            min_segment, max_segment = min(task_qa['segment_id']), max(task_qa['segment_id'])
            if max_segment-min_segment == 0:
                continue
            task_qa.pop('caption')
            task2qa[task].append(task_qa)
    return task2qa

def filter_qa_general(save_dir, stage, quality_judge = False):
    tasks = STAGE2TASK[stage]
    original_list = list()
    judgement_list = list()
    merged_list = list()
    avail_vids = load_json('./video_names.json')
    if stage == 'single':
        for task in tasks:
            task_judgements = parse_judgements(os.path.join(save_dir, task+'_modality_judgements.json'))
            task_judgements = [i for i in task_judgements if i['judgement'] == 'YES']
            judgement_list += task_judgements
            parsed_qa = parse_single_scene(os.path.join(save_dir, task+'.json'))
            parsed_qa = [i for i in parsed_qa if i['question'] != '' and i['answer'] != '' and i['explanation'] != '']
            original_list += parsed_qa
        df1 = pd.DataFrame(original_list)
        df2 = pd.DataFrame(judgement_list)
        merged_df = pd.merge(df1, df2, on=['video_name', 'segment_id', 'task'], how='inner')
    else:
        for task in tasks:
            parsed_qa = parse_multi_scene(os.path.join(save_dir, task+'.json'))
            parsed_qa = [i for i in parsed_qa if i['question'] != '' and i['answer'] != '' and i['explanation'] != '']
            original_list += parsed_qa
            task_judgements = parse_judgements(os.path.join(save_dir, task+'_modality_judgements.json'))
            task_judgements = [i for i in task_judgements if i['judgement'] == 'YES']
            judgement_list += task_judgements
        df1 = pd.DataFrame(original_list)
        df2 = pd.DataFrame(judgement_list)
        merged_df = pd.merge(df1, df2, on=['video_name', 'task', 'question'], how='inner')
    if quality_judge:
        quality_list = list()
        for task in tasks:
            task_judgements = parse_judgements(os.path.join(save_dir, task+'_quality_judgements.json'))
            task_judgements = [i for i in task_judgements if i['judgement'] == 'YES']
            quality_list += task_judgements
        quality_df = pd.DataFrame(quality_list)
        if stage == 'single':
            merged_df = pd.merge(merged_df, quality_df, on=['video_name', 'segment_id', 'task'], how='inner')
        else:
            merged_df = pd.merge(merged_df, quality_df, on=['video_name', 'task', 'question'], how='inner')
    merged_list = list()
    taskvidcount = defaultdict(dict)
    for item in merged_df.to_dict(orient='records'):
        taskvidcount[item['task']][item['video_name']] = taskvidcount[item['task']].get(item['video_name'], 0)
        merged_list.append({
            'qid':'_'.join((item['video_name'], item['task'], str(taskvidcount[item['task']][item['video_name']]))),
            'task':item['task'],
            'video_name':item['video_name'],
            'segment_id':item['segment_id'] if stage == 'single' else None,
            'question':item['question'],
            'answer':item['answer'],
            'explanation':item['explanation']
        })
        taskvidcount[item['task']][item['video_name']] += 1
        
    
    # from collections import defaultdict
    # segment_dict = defaultdict(list)
    # for caption in merged_list:
    #     vid_type = caption['type']
    #     segment_dict[vid_type].append(caption)
    # merged_list=list()
    # for vid_type in segment_dict.keys():
    #     import random
    #     print(len(segment_dict[vid_type]))
    #     merged_list += random.sample(segment_dict[vid_type], k=20)
                
    return merged_list

def correct_sequence(output_text, question, answer):
    # Initialize result dictionary
    result = {
        'status': None,
        'corrected_order': None,
        'explanation': None,
        'new_question': None,
        'new_answer': None
    }
    
    # Extract status
    status_match = re.search(r'\[(YES|NO|Corrected)\]', output_text)
    if status_match:
        result['status'] = status_match.group(1)
    
    # For Corrected cases, extract corrected order and explanation
    if result['status'] == 'Corrected':
        # Extract corrected order
        corrected_match = re.search(r'\[Corrected:\s*(\([a-z]\)\s*)+\]', output_text)
        if corrected_match:
            result['corrected_order'] = corrected_match.group(0).strip()
        
        # Extract explanation
        explanation_match = re.search(r'\[Explanation\](.*?)(?=\n\n|\Z)', output_text, re.DOTALL)
        if explanation_match:
            result['explanation'] = explanation_match.group(1).strip()
    
    # For YES and Corrected cases, generate new randomized question
    if result['status'] in ['YES', 'Corrected']:
        # Extract elements from question
        elements = re.findall(r'\(([a-z])\)\s*(.*?)(?=\s*\([a-z]\)|$)', question)
        if not elements:
            elements = re.findall(r'\(([a-z])\)\s*([^(\n]+?)(?=\s*\([a-z]\)|$)', question)
        
        # Extract current correct order (from corrected or original answer)
        if result['status'] == 'Corrected':
            correct_order = re.findall(r'\(([a-z])\)', result['corrected_order'])
        else:
            correct_order = re.findall(r'\(([a-z])\)', answer)
        
        # Create mapping of option to element
        option_map = {opt: elem for opt, elem in elements}
        
        # Create new random order (shuffle while maintaining correct sequence)
        options = [opt for opt, elem in elements]
        correct_sequence = [option_map[opt] for opt in correct_order]
        
        # Generate all possible permutations that maintain the correct sequence
        # We'll shuffle the options but keep the elements in correct order
        shuffled_options = options.copy()
        random.shuffle(shuffled_options)
        
        # Create new mapping between shuffled options and elements
        new_mapping = {shuffled_options[i]: correct_sequence[i] for i in range(len(shuffled_options))}
        
        # Build new question
        new_question_parts = []
        for opt in sorted(shuffled_options):  # Sort to maintain (a), (b), (c) order in display
            new_question_parts.append(f'({opt}) {new_mapping[opt]}')
        new_question = question.split('?')[0] + '? ' + ' '.join(new_question_parts)
        
        # Build new answer (in correct order based on shuffled options)
        new_answer = ' '.join([f'({opt})' for opt in shuffled_options])
        
        result['new_question'] = new_question
        result['new_answer'] = new_answer
    
    return result

def filter_qa_specific(task2qa, save_dir = './results'):
    # 加载原始问题
    audio_check = {
        'speech_emotion':['task2', 'task6'],
        'sound_event':['task4', 'task5'],
        'music':['task7', 'task16'],
    }
    sequence_check = ['task8', 'task10']
    ambiguity_check = ['task2', 'task4', 'task5']
    
    # audio check
    for audio_type, tasks in audio_check.items():
        for task in tasks:
            checked_qas = load_json(os.path.join(save_dir, f'{task}_audio_{audio_type}.json'))
            quality_dict = {item['qid']: item for item in checked_qas}
    
            filtered_data = []
            
            for data_item in task2qa[task]:
                qid = data_item['qid']
                # 检查qid是否存在于quality_dict中
                if qid in quality_dict:
                    # 假设合格标记的键名为'is_valid'，根据实际情况调整
                    if quality_dict[qid].get('judgement', 'NO') == 'YES':
                        filtered_data.append(data_item)
            task2qa[task] = filtered_data
            
    # sequence_check
    for task in sequence_check:
        checked_qas = load_json(os.path.join(save_dir, f'{task}_sequence.json'))
        quality_dict = {item['qid']: item for item in checked_qas}
        filtered_data = []
        for data_item in task2qa[task]:
            qid = data_item['qid']
            # 检查qid是否存在于quality_dict中
            if qid in quality_dict:
                results = correct_sequence(quality_dict[qid].get('judgement_text', ''), data_item['question'], data_item['answer'])
                if results['status'] == 'YES' or results['status'] == 'Corrected':
                    data_item['question'] = results['new_question']
                    data_item['answer'] = results['new_answer']
                    if results['status'] == 'Corrected':
                        data_item['explanation'] =results['explanation']
                    filtered_data.append(data_item)
        task2qa[task] = filtered_data
    # ambiguity check
    for task in ambiguity_check:
        checked_qas = load_json(os.path.join(save_dir, f'{task}_ambiguity.json'))
        quality_dict = {item['qid']: item for item in checked_qas}

        filtered_data = []
        
        for data_item in task2qa[task]:
            qid = data_item['qid']
            # 检查qid是否存在于quality_dict中
            if qid in quality_dict:
                # 假设合格标记的键名为'is_valid'，根据实际情况调整
                if quality_dict[qid].get('judgement', 'NO') == 'YES':
                    filtered_data.append(data_item)
        task2qa[task] = filtered_data
    return task2qa

def check_word_count(correct_answer, options, max_diff=10):
    correct_words = len(correct_answer.split())
    max_diff = max(max_diff, correct_words)
    overall_words = correct_words
    
    for option in options:
        option_words = len(option.split())
        overall_words+=option_words
        if abs(option_words - correct_words) > max_diff:
            return False
        
    if overall_words > 100:
        return False
    return True

def process_segment_relative_times(data, vid2segments):
    processed_data = data.copy()
    vid_name = data["video_name"]
    segment2time = vid2segments[vid_name]
    # 获取当前片段的起始时间（从segment_id字段）
    current_segment_id = data["segment_id"]
    if current_segment_id is None:
        current_segment_id = 0
    else:
        current_segment_id = min(current_segment_id)

    segment_start = segment2time[current_segment_id][0]
    
    def replace_with_relative_time(text: str) -> str:
        """将文本中的Segment X替换为（时间 - 当前片段起始时间）"""
        if not isinstance(text, str):
            return text
            
        # 匹配Segment 1, Segment 2等（不区分大小写）
        segments = re.findall(r'Segment\s+([0-9]+)', text, flags=re.IGNORECASE)
        for seg in segments:
            seg = int(seg)
            absolute_start_time = segment2time[seg][0]  # 取start_time
            absolute_end_time = segment2time[seg][1]  # 取start_time
            relative_start_time = seconds2timestamp(max(absolute_start_time - segment_start, 0))
            relative_end_time = seconds2timestamp(max(absolute_end_time - segment_start, 0))
            time_str = f"{relative_start_time}-{relative_end_time}"  # 避免负数
            text = re.sub(
                fr'Segment\s+{seg}\b', 
                time_str, 
                text, 
                flags=re.IGNORECASE
            )
        return text
    # 处理所有字段
    for key in ['question', 'correct_answer', 'options', 'explanation']:
        if key in processed_data:
            if isinstance(processed_data[key], list):
                processed_data[key] = [replace_with_relative_time(opt) for opt in processed_data[key]]
            else:
                processed_data[key] = replace_with_relative_time(processed_data[key])
    
    return processed_data

def extract_distractors(text, remove_brackets=True):
    distractor_pattern = r'\[Distractor \d+\]\s*(.*)'
    distractors = re.findall(distractor_pattern, text)
    distractor_list = list()
    if remove_brackets:
        for distractor in distractors:
            cleaned_distractor = re.sub(r'\[.*?\]', '', distractor).strip()
            if cleaned_distractor:
                distractor_list.append(cleaned_distractor)
    else:
        distractor_list = distractors
    return distractor_list

def replace_task13(item):
    match = re.search(r'segment\s+(\d+)', item['question'], re.IGNORECASE)
    new_item = item.copy()
    status = False
    if match:
        segment_num = int(match.group(1))
        # Update the item
        new_item["segment_id"] = [0, segment_num-1]
        new_item["question"] = "Which of the following options is most likely to occur after this video ends?"
        status = True
    
    return status, new_item

def replace_task9(item):
    pattern1 = "Which dialogue in other segments is most relevant to what the"
    pattern2 = "Which conversation in other clips is most related to the"
        
    question = item["question"]
    processed_item = None
    if question.startswith(pattern1) or question.startswith(pattern2):
        # 替换两种可能的短语
        new_question = question.replace("in other segments", "in the remaining parts")
        new_question = new_question.replace("in other clips", "in the remaining parts")
        
        # 创建新字典保留原数据
        processed_item = item.copy()
        processed_item["question"] = new_question

    if processed_item is not None:
        return True, processed_item
    else:
        return False, item

def time_formatter(segment_id, vid_segments, time_format):
    if isinstance(segment_id, int):
        start_segment, end_segment = segment_id, segment_id
    elif isinstance(segment_id, list):
        start_segment, end_segment = min(segment_id), max(segment_id)
    else:  # None
        start_segment, end_segment = 0, -1
    start_time = vid_segments[start_segment][0]
    end_time = vid_segments[end_segment][-1]
    if time_format == 'timestamp':
        return [seconds2timestamp(start_time), seconds2timestamp(end_time)]
    else:
        return [start_time, end_time]

def gather_mcq(save_dir, stage, time_format = 'second'):
    vid2segments = load_json('./vid2timecode.json')

    avail_list = list()
    failed_list = list()
    for task in STAGE2TASK[stage]:
        distractors = load_json(os.path.join(save_dir, f'{task}_distractors.json'))
        for distractor in distractors:
            distractor['options'] = extract_distractors(distractor['distractors_text'])
            if len(distractor['options']) == 3 and len(set(distractor['options'])) == 3:
                if distractor['correct_answer'] not in distractor['options']:
                    # if not check_word_count(distractor['correct_answer'], distractor['options']):
                    #     distractor['failed_reason'] = 'word count failed'
                    #     failed_list.append(distractor)
                    #     continue
                    if task == 'task9':
                        status, distractor = replace_task9(distractor)
                        if not status:
                            distractor['failed_reason'] = 'replacement failed'
                            failed_list.append(distractor)
                            continue
                    if task == 'task13':
                        status, distractor = replace_task13(distractor)
                        if not status:
                            distractor['failed_reason'] = 'replacement failed'
                            failed_list.append(distractor)
                            continue
                    if stage in ['multi', 'full']:
                        distractor = process_segment_relative_times(distractor, vid2segments)
                    distractor['options'].append(distractor['correct_answer'])
                    distractor.pop('distractors_text')
                    if time_format != 'id':
                        distractor['segment_id'] = time_formatter(distractor['segment_id'], vid2segments[distractor['video_name']], time_format = time_format)
                    avail_list.append(distractor)
            else:
                distractor['failed_reason'] = 'option num or uniqueness'
                failed_list.append(distractor)
                    
    return avail_list, failed_list


if __name__ == '__main__':
    # qa_data_path = './single_scene.json'
    # parse_single_scene(qa_data_path, './parsed_qas.xlsx')
    audio_caption = 'cleaned_data/audio_caption_cleaning_v1.json'
    audio_caption_data = load_json(audio_caption)

    speech_emotion = 'cleaned_data/speech_emotion_cleaning_v1.json'
    speech_emotion_data = load_json(speech_emotion)

    video_captions = load_json('./raw_data/video_caption.json')
    matched_results = []
    
    audio_caption_dict = {
        (item['video_name'], item['segment_id']): item for item in audio_caption_data
    }
    ser_caption_dict = {
        (item['video_name'], item['segment_id']): item for item in speech_emotion_data
    }
    # 遍历video_captions，查找匹配的audio_caption
    for video_item in video_captions:
        key = (video_item['video_name'], video_item['segment_id'])
        matched_results.append({
            'video_name': video_item['video_name'],
            'segment_id': video_item['segment_id'],
            'start_time': video_item['start_time'],
            'end_time': video_item['end_time'],
            'video_caption': video_item['caption'],
            'music': audio_caption_dict[key]['music'] if key in audio_caption_dict else "", 
            'sound_event': audio_caption_dict[key]['sound_event'] if key in audio_caption_dict else "",
            'speech_emotion': ser_caption_dict[key]['speech_emotion'] if key in ser_caption_dict else ''
        })
    save_json(matched_results, './paired_captions_v1.json')
    
    
