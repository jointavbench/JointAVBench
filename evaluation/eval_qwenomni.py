import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "5,6"
import torch
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from vision_process import process_vision_info
import librosa
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

model = Qwen2_5OmniModel.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B", 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    attn_implementation="flash_attention_2", 
    enable_audio_output=False
)

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
SYSTEM_PROMPT = """
You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.
""".strip()
sr = 16000
def evaluation(input_batch, modality):
    if modality == 'a' or modality == 'av':
        USE_AUDIO_IN_VIDEO = True
        audios = list()
    else:
        USE_AUDIO_IN_VIDEO = False
        audios = None
    conversations = list()
    for input_data in input_batch:
        prompt, file_path, segments, _ = input_data
        start_time, end_time = segments[0], segments[1]
        if modality == "v":
            conversation = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": file_path+'.mp4',
                         'segment_start':start_time, 'segment_end':end_time, "max_frames": 32},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
        elif modality == "a":
            conversation = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": file_path},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            audios.append(librosa.load(file_path+'.m4a', sr=sr)[0][int(start_time*sr):int(end_time*sr)])
        elif modality == "av":
            conversation = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": file_path+'.mp4',
                        #  'segment_start':start_time, 'segment_end':end_time,
                         "max_frames": 32},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            audios.append(librosa.load(file_path+'.m4a', sr=sr)[0][int(start_time*sr):int(end_time*sr)])
            
        conversations.append(conversation)

    # Preparation for inference
    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    images, videos = process_vision_info(conversations)
    # audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    # Inference: Generation of the output text and audio
    text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)

    # text = processor.batch_decode(text_ids[:,inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    texts = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    response = [text.split('\nassistant\n')[-1] for text in texts]
    
    return response
