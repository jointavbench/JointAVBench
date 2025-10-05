from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from vision_process import process_vision_info
from vllm import LLM, SamplingParams

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    # max_num_batched_tokens = 131072,
    dtype='bfloat16',
    limit_mm_per_prompt={"video": 1},
)

sampling_params = SamplingParams(
    # temperature=0.1,
    # top_p=0.001,
    # repetition_penalty=1.05,
    max_tokens=256,
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
# processor.tokenizer.padding_side = "left"

def evaluation(input_batch, modality):
    llm_inputs = list()
    generated_texts = list()
    for input_data in input_batch:
        prompt, file_path, segments, _ = input_data
        start_time, end_time = segments[0], segments[1]
        message=[
            {
                "role": "user",
                "content": [
                    {
                        "type":"video",
                        "video":file_path+'.mp4',
                        "max_frames": 32,
                        'segment_start':start_time, 
                        'segment_end':end_time,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        # Preparation for inference
        texts = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs.append({
            "prompt": texts,
            "multi_modal_data": mm_data,

            # FPS will be returned in video_kwargs
            "mm_processor_kwargs": video_kwargs,
        })
    outputs = llm.generate(llm_inputs, sampling_params=sampling_params, use_tqdm = False)
    for output in outputs:
        generated_texts.append(output.outputs[0].text)
    return generated_texts