import os
import json
import pickle
from datetime import datetime, timedelta
from collections import defaultdict


task2audform = {
    "task1":['speech'],
    "task2":['speech_emotion'],
    "task4":['sound_event'],
    "task5":['sound_event'],
    "task6":['speech_emotion'],
    "task7":['music'],
    "task8":['speech'],
    "task9":['speech'],
    "task10":['speech', 'speech_emotion'],
    "task11":['speech'],
    "task12":['sound_event','music'],
    "task13":['speech'],
    "task15":['speech','speech_emotion','sound_event','music'],
    "task16":['music'],
    "task17":['speech'],
}

def load_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每一行的 JSON 对象
            json_obj = json.loads(line.strip())
            data_list.append(json_obj)
    return data_list

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def timestamp2seconds(timestamp):
    time_obj = datetime.strptime(timestamp, "%H:%M:%S.%f")
    # total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
    # return total_seconds
    delta = timedelta(hours=time_obj.hour, minutes=time_obj.minute, seconds=time_obj.second, microseconds=time_obj.microsecond)
    return delta.total_seconds()

def load_sub(sub_dir):
    files = os.listdir(sub_dir)
    vid2sub = dict()
    for file in files:
        transcription = load_json(os.path.join(sub_dir, file))
        segments = transcription['segments']
        vid_name = file.split('.')[0]
        vid2sub[vid_name] = segments
    return vid2sub

def load_caption(file_path):
    data = load_json(file_path)
    transformed_data = defaultdict(list)
    for item in data:
        video_name = item['video_name']
        # segment_id = {k:v for k,v in item if k != 'video_name'}
        transformed_data[video_name].append(item)
    return transformed_data

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

def join_subtitles(subtitles, start_time = 0):
    joined_sub = ''
    for subtitle in subtitles:
        joined_sub += f"From {max(subtitle['start']-start_time,0):.2f} to {subtitle['end']-start_time:.2f}: {subtitle['text'].strip()}\n"
    return joined_sub

def merge_captions_segment_wise(video_captions, subtitles, aud_forms, segment_id=[], vid_desc=None):
    all_caps = ''
    if vid_desc is not None:
        all_caps += f"Story Description: {vid_desc}\n"
    if segment_id:
        if isinstance(segment_id, int):
            # Process only the specific segment_id
            target_segment_id = segment_id
            for vc in video_captions:
                if vc['segment_id'] == target_segment_id:
                    vc_start_time = timestamp2seconds(vc['start_time'])
                    vc_end_time = timestamp2seconds(vc['end_time'])
                    all_caps += f"From {vc_start_time-vc_start_time:.2f} to {vc_end_time-vc_start_time:.2f}:\n"
                    if 'speech' in aud_forms:
                        segment_subtitle = join_subtitles([sub for sub in subtitles if not (sub['end'] <= vc_start_time or sub['start'] >= vc_end_time)], vc_start_time)
                        all_caps += f"Subtitles: {segment_subtitle}\n" if segment_subtitle != '' else ''
                    if 'speech_emotion' in aud_forms:
                        all_caps += f"Speech Emotion: {vc['speech_emotion']}\n" if vc['speech_emotion'] != '' else ''
                    if 'sound_event' in aud_forms:
                        all_caps += f"Sound Event: {vc['sound_event']}\n" if vc['sound_event'] != '' else ''
                    if 'music' in aud_forms:
                        all_caps += f"Music: {vc['music']}\n" if vc['music'] != '' else ''
                    break  # Exit loop after processing the target segment
        elif isinstance(segment_id, list) and len(segment_id) == 2:
            # Process from the first segment_id to the second segment_id
            start_segment_id, end_segment_id = min(segment_id), max(segment_id)
            for vc in video_captions:
                if vc['segment_id'] == start_segment_id:
                    segment_start_time = timestamp2seconds(vc['start_time'])
            for vc in video_captions:
                if start_segment_id <= vc['segment_id'] <= end_segment_id:
                    vc_start_time = timestamp2seconds(vc['start_time'])
                    vc_end_time = timestamp2seconds(vc['end_time'])
                    aud_caps = ''
                    if 'speech' in aud_forms:
                        segment_subtitle = join_subtitles([sub for sub in subtitles if not (sub['end'] <= vc_start_time or sub['start'] >= vc_end_time)],segment_start_time)
                        aud_caps += f"Subtitles: {segment_subtitle}\n" if segment_subtitle != '' else ''
                    if 'speech_emotion' in aud_forms:
                        aud_caps += f"Speech Emotion: {vc['speech_emotion']}\n" if vc['speech_emotion'] != '' else ''
                    if 'sound_event' in aud_forms:
                        aud_caps += f"Sound Event: {vc['sound_event']}\n" if vc['sound_event'] != '' else ''
                    if 'music' in aud_forms:
                        aud_caps += f"Music: {vc['music']}\n" if vc['music'] != '' else ''
                    if aud_caps == '':
                        continue
                    else:
                        all_caps += f"From {vc_start_time-segment_start_time:.2f} to {vc_end_time-segment_start_time:.2f}:\n{aud_caps}"
                       
    else:
        # Process all video_captions
        for vc in video_captions:
            aud_caps = ''
            if 'speech' in aud_forms:
                segment_subtitle = join_subtitles([sub for sub in subtitles if not (sub['end'] <= timestamp2seconds(vc['start_time']) or sub['start'] >= timestamp2seconds(vc['end_time']))])
                aud_caps += f"Subtitles: {segment_subtitle}\n" if segment_subtitle != '' else ''
            if 'speech_emotion' in aud_forms:
                aud_caps += f"Speech Emotion: {vc['speech_emotion']}\n" if vc['speech_emotion'] != '' else ''
            if 'sound_event' in aud_forms:
                aud_caps += f"Sound Event: {vc['sound_event']}\n" if vc['sound_event'] != '' else ''
            if 'music' in aud_forms:
                aud_caps += f"Music: {vc['music']}\n" if vc['music'] != '' else ''
            if aud_caps == '':
                continue
            else:
                all_caps += f"From {timestamp2seconds(vc['start_time']):.2f} to {timestamp2seconds(vc['end_time']):.2f}:\n{aud_caps}"
    
    # all_caps += f'Subtitles:\n{subtitles}'
    return all_caps