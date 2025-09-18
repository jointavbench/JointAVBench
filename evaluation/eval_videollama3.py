import torch
from transformers import AutoModelForCausalLM, AutoProcessor

def load_model(device="cuda"):
    model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

def evaluation(input_data, modality, model_conf, device):
    model, processor = model_conf
    prompt, file_path, segments, _ = input_data
    video_path = file_path+'.mp4' 
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "max_frames": 32}},
                {"type": "text", "text": prompt},
            ]
        },
    ]

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=256)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response