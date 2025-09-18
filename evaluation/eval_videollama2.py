import sys
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
from transformers import logging
logging.set_verbosity_error()

def load_model(device):
    """Helper function to load a model instance for a specific device"""
    return model_init('DAMO-NLP-SG/VideoLLaMA2.1-7B-AV', device=device, num_frames = 32)


def evaluation(input_data, modality, model_conf, device):
    model, processor, tokenizer = model_conf
    prompt, file_path, segments, _ = input_data
    if modality == "a":
        model.model.vision_tower = None
    elif modality == "v":
        model.model.audio_tower = None
    elif modality == "av":
        pass
    else:
        raise NotImplementedError
    
    preprocess = processor['audio' if modality == "a" else "video"]
    if modality == "a":
        audio_video_tensor = preprocess(file_path+'.wav')
    else:
        audio_video_tensor = preprocess(file_path+'.mp4', s = segments[0], e = segments[1], va=True if modality == "av" else False)
    output = mm_infer(
        audio_video_tensor,
        prompt,
        model=model,
        tokenizer=tokenizer,
        modal='audio' if modality == "a" else "video",
        do_sample=False,
        device = device,
    )
    return output