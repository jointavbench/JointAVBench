# Evaluation Guide for JointAVBench

This folder contains evaluation code for testing models on the JointAVBench benchmark.

> **Note**: The benchmark data (`jointavbench.json`) is publicly available. You can use the evaluation scripts provided here, or write your own custom evaluation code to test models on our benchmark. The data format is straightforward and documented in the main [README](../README.md).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Running Evaluation](#running-evaluation)
- [Results Analysis](#results-analysis)

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for local models)
- 32GB+ RAM (recommended)
- **Models installed**: Install the models you want to evaluate according to their official repositories

## Quick Start

### 1. Download Dataset

```bash
# Download benchmark videos
pip install huggingface_hub
huggingface-cli download JointAVBench/JointAVBench --local-dir ../data
```

The benchmark questions file `jointavbench.json` is in the repository root. Please also download the video according to the 'video_url' column in the benchmark file.

### 2. Configure Environment

Create a `.env` file in the `evaluation/` directory:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# API Keys (for API-based models)
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
GEMINI_API_KEY=your_gemini_key_here

# Model paths (for local models)
VITA_MODEL_PATH=/path/to/vita-model
KIMI_AUDIO_MODEL_PATH=/path/to/kimi-audio-model
QWEN_VL_MODEL_PATH=/path/to/qwen-vl-model
INTERNVL_MODEL_PATH=/path/to/internvl-model

# Data paths
VIDEO_DIR=../data/videos
```

### 3. Run Evaluation

```bash
python evaluation.py \
    --qa-path ../jointavbench.json \
    --video-dir ../data/videos \
    --model-name <model_name> \
    --modality av
```

## Supported Models

We support 18 models. Use the `--model-name` argument with these exact names:

### API-based Models (3 models)
| Model Name | Description | Requirements |
|------------|-------------|--------------|
| `gemini` | Gemini Pro Vision | GEMINI_API_KEY |
| `gemini_api` | Gemini API version | GEMINI_API_KEY |
| `gpt4o` | GPT-4 with vision | OPENAI_API_KEY |

### Open-source Models
| Model Name | Description | Model Path Variable | Notes |
|------------|-------------|---------------------|-------|
| `qwen2.5vl` | Qwen 2.5 Vision-Language | QWEN_VL_MODEL_PATH | Batch processing |
| `qwen2.5omni` | Qwen 2.5 Omni (audio+vision) | QWEN_MODEL_PATH | Batch processing |
| `qwen2audio` | Qwen 2 Audio | QWEN_AUDIO_MODEL_PATH | Batch processing |
| `kimiaudio` | Kimi Audio | KIMI_AUDIO_MODEL_PATH | Multiprocessing |
| `llavavideo` | LLaVA Video | Model path in eval script | Multiprocessing |
| `internvl` | InternVL | INTERNVL_MODEL_PATH | Multiprocessing |
| `videollama2` | Video-LLaMA 2 | VIDEOLLAMA_MODEL_PATH | Multiprocessing |
| `videollama3` | Video-LLaMA 3 | VIDEOLLAMA_MODEL_PATH | Multiprocessing |
| `vita1.5` | VITA 1.5 | VITA_MODEL_PATH | Multiprocessing |
| `omnir1` | Omni R1 | Model path in eval script | Multiprocessing |
| `avicuna` | Audio-Visual Vicuna | Model path in eval script | Multiprocessing |
| `salmonn` | SALMONN | - | Requires pre-generated results |
| `salmonno1` | SALMONN O1 | - | Requires pre-generated results |
| `onellm` | OneLLM | - | Requires pre-generated results |
| `aurelia` | Aurelia | - | Requires pre-generated results |

**Notes**:
- **Batch/Multiprocessing models**: Install from official repositories and set model paths in `.env`
- **Pre-generated results models** (salmonn, salmonno1, onellm, aurelia): Run inference separately and provide result JSON files as shown in evaluation.py
- All model-specific dependencies should be installed according to their official documentation

## Running Evaluation

### Command Structure

```bash
python evaluation.py \
    --qa-path <path_to_jointavbench.json> \
    --video-dir <video_directory> \
    --model-name <model_name> \
    --modality <modality>
```

### Arguments

**Required:**
- `--qa-path`: Path to benchmark JSON file (usually `../jointavbench.json`)
- `--model-name`: Model to evaluate (see [Supported Models](#supported-models))

**Optional:**
- `--video-dir`: Directory containing video files (default: `../data/videos`)
- `--modality`: Input modality (default: `av`)
  - `a` - Audio only
  - `v` - Video only
  - `av` - Audio + Video (default)

### Examples

#### API-based Models

```bash
# Gemini
python evaluation.py \
    --qa-path ../jointavbench.json \
    --video-dir ../data/videos \
    --model-name gemini \
    --modality av

# GPT-4o
python evaluation.py \
    --qa-path ../jointavbench.json \
    --video-dir ../data/videos \
    --model-name gpt4o \
    --modality av
```

#### Local Models

```bash
# Qwen 2.5 VL
python evaluation.py \
    --qa-path ../jointavbench.json \
    --video-dir ../data/videos \
    --model-name qwen2.5vl \
    --modality av

# VITA 1.5
python evaluation.py \
    --qa-path ../jointavbench.json \
    --video-dir ../data/videos \
    --model-name vita1.5 \
    --modality av

# InternVL
python evaluation.py \
    --qa-path ../jointavbench.json \
    --video-dir ../data/videos \
    --model-name internvl \
    --modality av
```

#### Different Modalities

```bash
# Audio only
python evaluation.py \
    --qa-path ../jointavbench.json \
    --model-name qwen2audio \
    --modality a

# Video only
python evaluation.py \
    --qa-path ../jointavbench.json \
    --video-dir ../data/videos \
    --model-name qwen2.5vl \
    --modality v
```

### Batch Evaluation

Evaluate multiple models with a script:

```bash
#!/bin/bash
models=("gemini" "qwen2.5vl" "internvl" "vita1.5")

for model in "${models[@]}"
do
    echo "Evaluating $model..."
    python evaluation.py \
        --qa-path ../jointavbench.json \
        --video-dir ../data/videos \
        --model-name $model \
        --modality av
done
```

## Results Analysis

### Output Format

Results are saved to `./results/eval_results_<model>_<modality>.jsonl`

Each line contains:
```json
{
  "question_id": "...",
  "model_answer": "A",
  "correct_answer": "B",
  "model_output": "...",
  "task_type": "STL",
  "cognitive_dimension": "temporal",
  "audio_type": "speech",
  "scene_span": "single"
}
```

### Analyze Results

```bash
python eval_result.py \
    --results_dir ./results \
    --output_path ./results/analysis.json
```

This generates:
- Overall accuracy for each model
- Per-task accuracy breakdown
- Per-cognitive-dimension performance
- Per-audio-type performance
- Per-scene-span performance

## Custom Evaluation

You can write your own evaluation code instead of using our scripts. The benchmark data is publicly available in `../jointavbench.json`.

### Data Format

Each question has the following structure:

```json
{
  "qid": "unique_question_id",
  "video_name": "video_identifier",
  "task": "STL",
  "question": "Question text",
  "correct_answer": "Correct answer text",
  "explanation": "Explanation of the answer",
  "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
  "video_url": "https://www.youtube.com/watch?v=...",
  "segment_timestamp": [start_seconds, end_seconds]
}
```

### Custom Evaluation Steps

1. **Load benchmark data**: Read `jointavbench.json`
2. **Load videos**: Download from Hugging Face or use `video_url`
3. **Get model predictions**: Run your model on each question with video input
4. **Compare answers**: Match model output against `correct_answer`
5. **Calculate metrics**: Compute accuracy overall and by task/dimension/scene span

### Example Custom Evaluation Code

```python
import json

# Load benchmark
with open('jointavbench.json', 'r') as f:
    benchmark = json.load(f)

# Evaluate your model
correct = 0
total = len(benchmark)

for item in benchmark:
    video_path = f"data/videos/{item['video_name']}.mp4"
    question = item['question']
    options = item['options']

    # Run your model (implement this)
    model_answer = your_model.predict(video_path, question, options)

    # Check correctness
    if model_answer == item['correct_answer']:
        correct += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.2%}")
```

Feel free to implement your own evaluation pipeline that suits your model architecture and requirements!
