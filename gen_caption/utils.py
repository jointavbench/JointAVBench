import os
import pickle
import cv2
import json
import pandas as pd
from datetime import datetime, timedelta
from scenedetect import detect, ContentDetector, open_video, SceneManager
from huggingface_hub import hf_hub_url, file_exists
from tqdm import tqdm
import shutil
from pathlib import Path
from collections import defaultdict



def save_json(data, file_path):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)        

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

def jsonl2json(jsonl_file_path, output_json_path=None):
    # 读取JSONL文件
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # 排序函数：先按video_name，再按segment_id
    def sort_key(item):
        return (item.get('video_name', ''), item.get('segment_id', 0))
    
    # 对数据进行排序
    sorted_data = sorted(data, key=sort_key)
    
    # 如果需要保存到文件
    if output_json_path:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, ensure_ascii=False, indent=2)
    
    return sorted_data

def json2xlsx(file_path, save_path):
    data = load_json(file_path)
    df = pd.DataFrame(data)
    df.to_excel(save_path, index=False)
    
def seconds2timestamp(seconds):
    t = timedelta(seconds=seconds)
    
    total_seconds = int(t.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = t.microseconds // 1000
    
    time_format = f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
    return time_format
    
def timestamp2seconds(timestamp):
    time_obj = datetime.strptime(timestamp, "%H:%M:%S.%f")
    # total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
    # return total_seconds
    delta = timedelta(hours=time_obj.hour, minutes=time_obj.minute, seconds=time_obj.second, microseconds=time_obj.microsecond)
    return delta.total_seconds()

def valid_hf_audio(file_path):
    vid2timecode= load_json(file_path)
    video_names = sorted(list(vid2timecode.keys()))
    valid_audios = list()
    for video_name in tqdm(video_names):
        for segment_id, (start_time, end_time) in enumerate(vid2timecode[video_name]):
            if file_exists("roverx12345/sfd_audio_split", f"{video_name}_{segment_id}.wav", repo_type = "dataset"):
                valid_audios.append([video_name, segment_id])
    save_json(valid_audios, 'valid_audios.json')
    
def split_files_into_folders(source_dir, files_per_folder=8000, target_base_name="split_folder"):
    """
    将源文件夹中的文件分割到多个子文件夹中
    
    参数:
        source_dir (str): 源文件夹路径
        files_per_folder (int): 每个子文件夹中的文件数量
        target_base_name (str): 子文件夹的基础名称
    """
    source_path = Path(source_dir)
    
    # 获取源文件夹中所有文件（不包括子文件夹）
    all_files = [f for f in source_path.iterdir() if f.is_file()]
    total_files = len(all_files)
    
    print(f"找到 {total_files} 个文件，将分割为 {files_per_folder} 个文件每文件夹")
    
    # 计算需要创建多少个子文件夹
    num_folders = total_files // files_per_folder
    if total_files % files_per_folder != 0:
        num_folders += 1
    
    # 创建子文件夹并移动文件
    for folder_num in range(num_folders):
        start_idx = folder_num * files_per_folder
        end_idx = start_idx + files_per_folder
        folder_files = all_files[start_idx:end_idx]
        
        # 创建目标文件夹
        target_folder = source_path / f"{target_base_name}_{folder_num + 1}"
        target_folder.mkdir(exist_ok=True)
        
        # 移动文件
        for file in folder_files:
            shutil.move(str(file), str(target_folder / file.name))
        
        print(f"已创建 {target_folder.name} 包含 {len(folder_files)} 个文件")
    
    print(f"\n分割完成！共创建 {num_folders} 个子文件夹")

def create_file_folder_mapping(root_dir):
    """
    创建文件名到文件夹名称的映射字典
    
    参数:
        root_dir (str): 根文件夹路径
        
    返回:
        dict: {文件名: 文件夹名} 的字典
        dict: {文件夹名: [文件列表]} 的字典
    """
    root_path = Path(root_dir)
    
    # 验证路径是否存在且是目录
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"提供的路径不存在或不是文件夹: {root_dir}")
    
    # 初始化两个字典
    file_to_folder = {}      # 文件名 -> 文件夹名
    folder_to_files = defaultdict(list)  # 文件夹名 -> 文件列表
    
    # 遍历所有子文件夹
    for folder in root_path.iterdir():
        if folder.is_dir():
            folder_name = folder.name
            
            # 遍历子文件夹中的文件
            for file in folder.iterdir():
                if file.is_file():
                    file_name = file.name
                    file_to_folder[file_name] = folder_name
                    folder_to_files[folder_name].append(file_name)
    
    return file_to_folder, dict(folder_to_files)

class PySceneSegmenter:
    def __init__(self, video_path='', 
                 threshold=27.0, 
                 min_scene_len = 5, 
                 save_path = './scene.pkl', 
                 ):
        """
        初始化视频分割迭代器类。

        :param video_path: 输入视频文件路径。
        :param threshold: 内容监测器判定分割的阈值（越高越少分割，越低越多）。
        :param min_scene_len: PySceneDetect分割视频的最小时长(秒)
        :param output_path: 分割结果保存csv文件
        """
        self.video_path = video_path
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.scene_list = []
        self.save_path = save_path
        
        if not os.path.exists(save_path):
            self._detect_scenes()
        else:
            self._load_scenes()
        
    @staticmethod
    def video_probe(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Error: Could not open video.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return fps, total_frames

    def _write_scene_list(self):
        with open(self.save_path, 'wb') as file:
            pickle.dump(self.scene_list, file)
        
    def _detect_scenes(self):
        fps, total_frames = self.video_probe(self.video_path)
        video_stream = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.auto_downscale = True
        scene_manager.add_detector(ContentDetector(threshold=self.threshold, min_scene_len = int(self.min_scene_len*fps)))
        scene_manager.detect_scenes(video_stream)
        self.scene_list = scene_manager.get_scene_list()
        self._write_scene_list()

    def _load_scenes(self):
        with open(self.save_path, 'rb') as file:
            self.scene_list = pickle.load(file)

    def __len__(self):
        return len(self.scene_list)
        
    def __iter__(self):
        return self

    def __next__(self):
        if not self.scene_list:
            raise StopIteration
        start_time, end_time = self.scene_list.pop(0)
        return str(start_time), str(end_time)