# Adopted from https://github.com/huangb23/VTimeLLM
import os
import sys
import torch
import laion_clap
from avicuna.constants import IMAGE_TOKEN_INDEX
from avicuna.conversation import conv_templates, SeparatorStyle
from avicuna.model.builder import load_pretrained_model, load_lora
from avicuna.utils import disable_torch_init
from avicuna.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
from avicuna.test_clap import audio_feat_extraction
from PIL import Image
import requests
import librosa
from io import BytesIO
from transformers import TextStreamer
from easydict import EasyDict as edict
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import numpy as np
import clip

def load_model(device="cuda"):
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model([], "AVicuna/checkpoints/avicuna-vicuna-v1-5-7b-stage3", "AVicuna/checkpoints/avicuna-vicuna-v1-5-7b-stage4")
    model = model.cuda(device)
    model.to(torch.bfloat16)
    
    clip_model, _ = clip.load("AVicuna/checkpoints/clip/ViT-L-14.pt")
    clip_model.eval()
    clip_model = clip_model.cuda(device)
    
    clap_model = laion_clap.CLAP_Module(enable_fusion=True).to(device)
    clap_model.load_ckpt("AVicuna/checkpoints/630k-fusion-best.pt")
    return tokenizer, model, clip_model, clap_model


def inference(model, image, query, tokenizer, device):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda(device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image,
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


def evaluation(input_data, modality, model_conf, device):
    tokenizer, model, clip_model, clap_model = model_conf
    prompt, file_path, segments, _ = input_data
    av_ratio = 0.25
    n_audio_feats = int(100 * av_ratio)
    n_image_feats = int(100 - n_audio_feats)

    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    video_loader = VideoExtractor(N=n_image_feats)
    _, images = video_loader.extract({'id': None, 'video': file_path+'.mp4'})
    # print(images.shape) # <N, 3, H, W>
    images = transform(images / 255.0)
    images = images.to(torch.bfloat16)
    
    with torch.no_grad():
        audio = audio_feat_extraction(file_path+'.wav', clap_model)
        v_features = clip_model.encode_image(images.to(device))
        a_features = audio.cuda(device)

        tmp_len = len(a_features)
        if tmp_len != n_audio_feats:
            repeat_factor = n_audio_feats // tmp_len
            remainder = n_audio_feats % tmp_len
            a_features = torch.cat([a_features[i].unsqueeze(0).repeat(repeat_factor + (1 if i < remainder else 0), 1) for i in range(tmp_len)], dim=0)
            # print(v_features.shape, a_features.shape)
        features = [v_features.unsqueeze(0), a_features.unsqueeze(0)]

    text = inference(model, features, "<video>\n " + prompt, tokenizer, device)
    return text