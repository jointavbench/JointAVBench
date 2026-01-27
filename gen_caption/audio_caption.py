import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import json
from tqdm import tqdm
import pickle
import librosa
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from utils import PySceneSegmenter, timestamp2seconds, load_json
import torch
import argparse

MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"
# MODEL_PATH = "/fs/fast/u2021201666/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/dc4921a57347e9f81190db97fd9f73166f2229ea/"
CAPTION_SAVE_PATH = 'prompt_result/prompt_v11.json'
TEST_SYSTEM_PROMPT = """
You are an expert in speech emotion recognition. Your task is to analyze and describe the content and tone of the speech without adding extra details, interpretations, or speculations.
""".strip()
TEST_CAPTION_PROMPT = """
nalyze the provided audio clip containing multiple dialogue segments. For each spoken dialogue segment, output ONLY the following confirmed information in a structured format:

Dialogue Content: [Transcribe the exact spoken words]
Speech Emotion: [Analyze ONLY vocal characteristics (tone, pitch, pace) to determine emotion - do NOT infer from content]
Speaker Traits: [Identify ONLY detectable traits from voice (gender, approximate age range)]
Requirements:

Ignore all non-dialogue audio elements
Only report confirmed observations from vocal analysis
Never infer emotions from dialogue content
If any element cannot be determined, omit it entirely
Maintain strict separation between vocal analysis and content analysis
""".strip()

AUD_SYSTEM_PROMPT = """
You are a highly precise and detail-oriented audio assistant. Your task is to describe sounds with accuracy, providing clear and objective descriptions based solely on the content. Avoid guesses and any form of ambiguous language.
""".strip()
# AUD_CAPTION_PROMPT = """
# Describe the audio by focusing **ONLY** on the following elements:  
# 1. Sound Events. Identify and describe:  
#    - Non-speech sounds: (e.g., crumpling, footsteps, door closing, glass breaking).  
#    - Non-verbal vocalizations: (e.g., laughter, sighing, coughing, crying, humming, screaming).  
#    - Characteristics:  
#      - Pitch (high/low), timbre (bright/muffled), rhythm (steady/erratic), volume (loud/soft).  
#    - Do NOT describe speech, spoken words, or conversation content.
# 2. Background Music:
#    - Melody(simple/complex, repetitive/varied).  
#    - Instruments (e.g., piano, strings, electric guitar).  
#    - Mood(e.g., cheerful, tense, melancholic).  
#    - Avoid technical terms (BPM, key, scales).  
# **Rules:**  
# - Never describe speech, spoken words, or verbal interactions.
# - Non-verbal vocalizations (e.g., laughter, sighs) are allowed.
# - Do not infer speaker demographics (e.g., "a child laughing" → just "laughter").
# - No timestamps, durations, or causal explanations (e.g., "someone crumpling paper" → just "a crumpling sound").
# """.strip()
AUD_CAPTION_PROMPT = """
Describe the audio by focusing **ONLY** on the following detectable elements:  
1. Sound Events (if clearly present):  
    - Non-speech sounds: (e.g., crumpling, footsteps, door closing, glass breaking).  
    - Non-verbal vocalizations: (e.g., laughter, sighing, coughing, crying, humming, screaming).  
    - Characteristics (only if unambiguous):  
        - Pitch (high/low), timbre (bright/muffled), rhythm (steady/erratic), volume (loud/soft).  
    - Rules:
        - Do NOT guess sound sources (e.g., no "paper crumpling" → just "a crumpling sound").
        - If uncertain, omit the detail entirely.
2. Background Music (if clearly present):
    - Instruments (e.g., piano, strings, electric guitar).
    - Mood(e.g., cheerful, tense, melancholic).
    - Avoid technical terms (BPM, key, scales).
    - Rules:
        - Do NOT describe lyrics or vocal melodies.
        - If the music is ambiguous (e.g., genre unclear), only state observable features.
Critical Constraints:
- Do NOT describe speech, spoken words, conversation content, or verbal interactions.
- Never use speculative language (e.g., "likely," "probably," "seems like").
- If a sound cannot be confidently identified, skip it.
- No timestamps, durations, or speaker demographics (e.g., "a child laughing" → just "laughter").
""".strip()

SER_SYSTEM_PROMPT = """
You are an expert in speech emotion recognition. Your task is to analyze and describe the content and tone of the speech without adding extra details, interpretations, or speculations.
"""

SER_CAPTION_PROMPT = """
Please analyze the provided speech audio and generate a strictly factual description following these requirements:
1. Output format for each utterance:
    Speech Content: [Exact dialogue content]
    Emotion: [Observed emotional tone] (eg. happy, sad, angry, fearful, surprised, disgusted, excited)
    Speaker traits: [Directly discernible characteristics like age/gender if evident from voice] 
2. Rules:
    - Only describe emotions clearly conveyed through vocal tone
    - Note speaker characteristics ONLY when immediately apparent from voice (e.g. "child-like voice")
    - Never add interpretations beyond what the audio contains
    - Process each utterance separately
    - Non-speech audio (music or sound only): output *"[Non-speech audio: skip analysis]"* and stop
"""
task2prompt = {
    'audio_caption':{
        'system': AUD_SYSTEM_PROMPT,
        'user': AUD_CAPTION_PROMPT
    },
    'speech_emotion':{
        'system': SER_SYSTEM_PROMPT,
        'user': SER_CAPTION_PROMPT
    },
    'test':{
        'system':TEST_SYSTEM_PROMPT,
        'user': TEST_CAPTION_PROMPT
    }
}
def batch_inputs(inputs, metadata, max_duration=512):
    current_batch = []  
    current_metadata = []
    current_duration = 0 
    for input_data, meta in zip(inputs,metadata):
        duration = meta[-1]
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

def prepare_inputs(audio_path, segment_save_path, task_name, coverage = None):
    inputs = list()
    metadata = list()

    segmenter = PySceneSegmenter(save_path=segment_save_path) 
    audio_content, sr = librosa.load(audio_path, sr=16000)
    segments_timestamp = [[start_time, end_time] for start_time, end_time in segmenter]
    segments = list()
    if coverage is not None:
        for covered_segment in coverage:
            start_time = segments_timestamp[covered_segment[0]][0]
            end_time = segments_timestamp[covered_segment[-1]][-1]
            segments.append([start_time, end_time])
    else:
        segments = segments_timestamp
    for idx, (start_time, end_time) in enumerate(segments):
        start_time = timestamp2seconds(start_time)
        end_time = timestamp2seconds(end_time)

        segment_duration = end_time - start_time
        
        # if round(segment_duration)>120:
        #     continue
        
        conversation = [
            {'role': 'system', 'content': task2prompt[task_name]['system']}, 
            {"role": "user", "content": [
                {"type": "audio", "audio": ''},
                {"type": "text", "text": task2prompt[task_name]['user']},
            ]},
        ]
        
        inputs.append((conversation, audio_content[int(start_time*sr):int(end_time*sr)]))
        metadata.append((idx, start_time, end_time, segment_duration))
            
    return inputs, metadata

def main(args):
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    model = Qwen2_5OmniModel.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="flash_attention_2", 
        enable_audio_output=False
    )
    task_name = args.task_name

    # Use arguments or environment variables for data paths
    directory_path = args.data_dir if hasattr(args, 'data_dir') and args.data_dir else os.environ.get('CAPTION_DATA_DIR', './data/captions')
    audio_directory = args.audio_dir if hasattr(args, 'audio_dir') and args.audio_dir else os.environ.get('AUDIO_DIR', './data/audio')
    video_names = sorted(os.listdir(directory_path))
    if args.merged:
        vid2coverage = load_json('./coverage.json')
        video_names = sorted(list(vid2coverage.keys()))
    audio_suffix = f'_{task_name}_omni_coverage.json' if args.merged else f'_{task_name}_omni.json'
    new_video_names = list()
    for video_name in video_names:
        processed_folder = os.path.join(directory_path, video_name)
        if not os.path.exists(processed_folder):
            continue
        if not os.path.exists(os.path.join(processed_folder, video_name+audio_suffix)):
            new_video_names.append(video_name)        
    video_names = new_video_names
    # ===========================
    # video_names = ['Y_b5wYLVmyw']
    # ===========================
    
    for video_name in tqdm(video_names):
        processed_folder = os.path.join(directory_path, video_name)
        audio_path = os.path.join(audio_directory, video_name+'.m4a')
        if not os.path.exists(audio_path):
            continue
        
        segment_save_path = os.path.join(processed_folder, video_name+'_scenes.pkl')
        caption_path = os.path.join(processed_folder, f'{video_name}{audio_suffix}')
        # ===========================
        # caption_path = CAPTION_SAVE_PATH
        # ===========================
        coverage = vid2coverage[video_name] if args.merged else None
        inputs, metadata = prepare_inputs(audio_path, segment_save_path, task_name = task_name, coverage = coverage)
        captions = list()
        for input_batch, meta_batch in batch_inputs(inputs, metadata):
            conversations = [x[0] for x in input_batch]
            audios = [x[1] for x in input_batch]
            text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text, audios=audios, images=None, videos=None, return_tensors="pt", padding=True, use_audio_in_video=True)
            inputs = inputs.to(model.device).to(model.dtype)
            text_ids = model.generate(**inputs, use_audio_in_video=True, return_audio=False, thinker_max_new_tokens = 512)
            
            generated_texts = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            generated_texts = [text.split('\nassistant\n')[-1] for text in generated_texts]
            for (segment_id, start_time, end_time, _), generated_text in zip(meta_batch, generated_texts):
                caption = {
                    'video_name': video_name,
                    'type': task_name,
                    'segment_id': segment_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'caption': generated_text,
                }
                captions.append(caption)

        try:
            with open(caption_path, 'w', encoding='utf-8') as f:
                json.dump(captions, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"An error occurred while writing to the file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name', choices=['audio_caption', 'speech_emotion', 'test'], help='', required=True)
    parser.add_argument('--merged', action = 'store_true', help='')
    parser.add_argument('--data-dir', type=str, default='./data/captions', help='Directory containing caption data')
    parser.add_argument('--audio-dir', type=str, default='./data/audio', help='Directory containing audio files')
    args = parser.parse_args()
    main(args)
