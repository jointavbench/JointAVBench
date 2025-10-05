import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from utils import load_caption, save_json, load_sub, prepare_sub, load_json
from parse_data import output_processing, clean_subtitles
from openai import OpenAI
from prompts import CLEAN_DATA_PROMPTS

MODEL_PATH = "Qwen/Qwen2.5-72B-Instruct"
BATCH_SIZE = 24
task = 'audio_caption_cleaning'
# Define the system prompt
SYSTEM_PROMPT = "You are a helpful assistant that performs data cleaning on audio captions, speech emotions, and subtitles. Your task is to follow my instructions."
API_KEY = "sk-d4099bd527ba48ba9d0fa6e58b35bfff"
MODEL_NAME = "qwen2.5-72b-instruct"

if __name__ == '__main__':
    caption_path = './raw_data/paired_captions_v1.json'
    sub_path = './raw_data/subtitle'
    save_path = './cleaned_data'
    
    paired_captions = load_caption(caption_path)
    vid2sub = load_sub(sub_path)
    
    vid_names = list(paired_captions.keys())
    vid_names.sort()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # client = OpenAI(
    #     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    #     api_key=API_KEY, 
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        dtype='bfloat16',
    )
    sampling_params = SamplingParams(
        temperature=0.1, 
        top_p=0.001, 
        repetition_penalty=1.05, 
        max_tokens=1024,
    )

    qas = list()
    result_save_path = os.path.join(save_path, f'{task}_v1.json')
    if os.path.exists(result_save_path):
        qas = load_json(result_save_path)
        video_names = list()
        for item in qas:
            video_names.append(item['video_name'])
        vid_names = [i for i in vid_names if i not in set(video_names)]
    
    # vid_names = ['Y_b5wYLVmyw']
    # result_save_path = './prompt_result/sec_prompt_v2.json'
    for vid in tqdm(vid_names, desc=f'Processing [{task}]'):
        vid_captions = paired_captions[vid]
        subtitles = vid2sub[vid]
        if len(subtitles) == 0 and task == 'speech_emotion_cleaning':
            continue
        subtitles = clean_subtitles(subtitles)
        vid_captions = prepare_sub(subtitles, vid_captions, False)
        texts = list()
        metadata = list()
        for caption in vid_captions:
            subtitle = caption.get('subtitle', '')
            if subtitle == '' and task == 'speech_emotion_cleaning':
                continue
            prompt_completion = {
                'audio_caption': caption.get('audio_caption', ''),
                'speech_emotion': caption.get('speech_emotion', ''),
                'subtitle': subtitle,
            }
            if prompt_completion['audio_caption'] == '' and prompt_completion['speech_emotion'] == "":
                continue
            segment_id, start_time, end_time, ac, se, sub = caption['segment_id'], caption['start_time'], caption['end_time'], prompt_completion['audio_caption'], prompt_completion['speech_emotion'], prompt_completion['subtitle']
            task_prompt = CLEAN_DATA_PROMPTS[task].format(**prompt_completion)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task_prompt}
            ]

            # completion = client.chat.completions.create(
            #     model=MODEL_NAME,
            #     messages=messages,
            #     temperature=0.1, 
            #     top_p=0.001, 
            #     presence_penalty =1.05, 
            #     max_tokens=1024,
            #     )
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            metadata.append((segment_id, start_time, end_time, ac, se, sub))
            texts.append(text)
        responses = list()
        for i in range(0, len(texts), BATCH_SIZE):
            responses += llm.generate(texts[i:i+BATCH_SIZE], sampling_params=sampling_params, use_tqdm=False)
        for (segment_id, start_time, end_time, ac, se, sub), response in zip(metadata, responses):
            generated_text = response.outputs[0].text
            # generated_text = completion.choices[0].message.content
            caption = {
                'video_name': vid,
                'type': task,
                'segment_id': segment_id,
                'start_time': start_time,
                'end_time': end_time,
                'caption': generated_text,
                'audio_caption':ac,
                'emotion_caption':se, 
                'subtitle':sub,
            }
            qas.append(output_processing(task, caption))
        save_json(qas, result_save_path)
        