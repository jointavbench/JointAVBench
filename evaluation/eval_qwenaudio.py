from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")
sr = 16000

def evaluation(input_batch, modality):
    if modality != "a":
        assert 1==0
    audios = list()
    conversations = list()
    for input_data in input_batch:
        prompt, file_path, segments, _ = input_data
        start_time, end_time = segments[0], segments[1]
        if end_time-start_time > 30:
            end_time = start_time + 30
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": file_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        audios.append(librosa.load(file_path+'.m4a', sr=sr)[0][int(start_time*sr):int(end_time*sr)])
        conversations.append(conversation)

    text = [processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False) for conversation in conversations]

    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs['input_ids'] = inputs['input_ids'].to("cuda")
    inputs.input_ids = inputs.input_ids.to("cuda")

    generate_ids = model.generate(**inputs, max_new_tokens=256)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return response