import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,4,5"
import torch
from transformers import (
    Qwen2_5OmniModel,
    Qwen2_5OmniProcessor,
    GenerationConfig,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_omni_utils import process_mm_info
from vision_process import process_vision_info
import librosa

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

SYSTEM_PROMPT = """
You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.
""".strip()
sr = 16000

def load_model(device):
    omni_path = "Haoz0206/Omni-R1"
    # Omni-R1 is Qwen2_5OmniThinker, not Qwen2_5OmniModel, so inference code is different from that of Qwen official codes.
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        omni_path,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(omni_path)

    generation_config = GenerationConfig(
        use_cache=True, max_new_tokens=1024, do_sample=False
    )
    return model, processor, generation_config

def evaluation(input_data, modality, model_conf, device):
    model, processor, generation_config = model_conf
    prompt, file_path, segments, _ = input_data
    start_time, end_time = segments[0], segments[1]
    conversation = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": file_path+'.mp4', "max_frames": 32},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text_input = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    audio_input, image_input, video_input, process_args = process_mm_info(
        conversation, use_audio_in_video=False
    )
    # images, videos = process_vision_info(conversations)
    inputs = processor(
        text=text_input,
        images=image_input,
        audios=audio_input,
        videos=video_input,
        return_tensors="pt",
        do_resize=True,
    ).to(device)
    # 生成输出
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, generation_config=generation_config)

    # Decode the generated completions
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    response = text[0].split('\nassistant\n')[-1]
    
    return response
