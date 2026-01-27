import os
import soundfile as sf
from kimia_infer.api.kimia import KimiAudio

def load_model(device, model_path=None):
    # Use environment variable or default model name
    if model_path is None:
        model_path = os.environ.get('KIMI_AUDIO_MODEL_PATH', 'moonshotai/Kimi-Audio-7B-Instruct')
    model = KimiAudio(model_path=model_path, load_detokenizer=False, device = device)
    return model

def evaluation(input_data, modality, model_conf, device):
    if modality !='a':
        assert 1==0, 'modality error!'
    model = model_conf
    prompt, file_path, segments, _ = input_data
    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
        "max_new_tokens":256,
    }

    # --- 3. Example 1: Audio-to-Text (ASR) ---
    messages_asr = [
        # You can provide context or instructions as text
        {"role": "user", "message_type": "text", "content": prompt},
        # Provide the audio file path
        {"role": "user", "message_type": "audio", "content": file_path+'.wav'}
    ]

    # Generate only text output
    _, text_output = model.generate(messages_asr, **sampling_params, output_type="text", device = device)
    return text_output