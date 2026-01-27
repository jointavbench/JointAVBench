# Benchmark Generation Pipeline

This module implements the automated generation pipeline for JointAVBench. It transforms audio-visual captions into challenging multiple-choice questions that require joint reasoning across both modalities.

## Overview

The pipeline takes timestamped audio-visual captions and generates questions through multiple stages:
1. **Interval Identification** - Find suitable video segments
2. **Question Generation** - Create questions requiring joint reasoning
3. **Distractor Generation** - Add plausible incorrect options
4. **Quality Control** - Validate question quality

## Files

### Core Generation Scripts

#### `identify_intervals.py`
**Purpose**: Identify suitable video segments (intervals) for each task type.

**What It Does**:
- Analyzes audio-visual captions to find relevant segments
- Determines which task types are suitable for each interval
- Considers scene boundaries and content requirements
- Ensures intervals have sufficient audio-visual information

**Key Features**:
- Uses LLM (Qwen2.5-72B) to analyze caption content
- Maps intervals to specific tasks (task1-task17)
- Handles single-scene, multi-scene, and full-video intervals
- Validates interval quality before question generation

**Usage**:
```bash
python identify_intervals.py \
    --caption_data /path/to/captions \
    --output_path ./intervals.json
```

**Output Format**:
```json
{
  "video_id": "example_video_001",
  "intervals": [
    {
      "interval_id": "interval_001",
      "start_time": "00:00:05.000",
      "end_time": "00:00:15.000",
      "task_types": ["task1", "task6"],
      "scene_type": "single",
      "visual_content": "Person speaking in office",
      "audio_content": "Speech with emotion",
      "reasoning": "Suitable for temporal and emotion tasks"
    }
  ]
}
```

**Task Assignment Logic**:
- **Single-scene tasks** (task1-8): Require one continuous scene
- **Multi-scene tasks** (task9-13): Require 2-3 scene transitions
- **Full-video tasks** (task15-17): Require entire video context

---

#### `generate_qa.py`
**Purpose**: Generate question-answer pairs from identified intervals.

**What It Does**:
- Takes intervals and their captions
- Uses LLM to generate questions requiring joint audio-visual reasoning
- Ensures questions cannot be answered with single modality
- Creates diverse question types across cognitive dimensions

**Key Features**:
- Prompt engineering to ensure multi-modal requirements
- Generates questions for each task type
- Creates ground-truth answers based on captions
- Validates that questions are grounded in evidence

**Usage**:
```bash
python generate_qa.py \
    --intervals ./intervals.json \
    --captions /path/to/captions \
    --output_path ./qa_pairs.json
```

**Question Generation Process**:
1. Load interval and its captions (audio + visual)
2. Select appropriate prompt template for task type
3. Generate question using LLM
4. Extract answer from captions
5. Validate question requires both modalities

**Output Format**:
```json
{
  "question_id": "q_001",
  "video_id": "example_video_001",
  "interval_id": "interval_001",
  "task_type": "task1",
  "question": "What is the temporal order of the speeches?",
  "answer": "The male voice speaks first, followed by the female voice",
  "cognitive_dimension": "temporal",
  "audio_type": "speech",
  "scene_span": "single"
}
```

---

#### `generate_distractor.py`
**Purpose**: Generate plausible but incorrect answer options (distractors).

**What It Does**:
- Takes QA pairs and generates 3 distractors per question
- Ensures distractors are plausible but clearly incorrect
- Balances difficulty across options
- Creates multiple-choice format (A/B/C/D)

**Key Features**:
- Uses LLM to generate contextually plausible distractors
- Avoids obviously wrong or nonsensical options
- Ensures distractors differ meaningfully from correct answer
- Randomizes option order

**Usage**:
```bash
python generate_distractor.py \
    --qa_path ./qa_pairs.json \
    --output_path ./final_qa.json
```

**Distractor Types**:
1. **Temporal confusion**: Wrong time ordering
2. **Content confusion**: Similar but incorrect content
3. **Modality confusion**: Correct audio but wrong visual (or vice versa)
4. **Logical negation**: Opposite of correct answer

**Output Format**:
```json
{
  "question_id": "q_001",
  "question": "What is the temporal order of the speeches?",
  "options": {
    "A": "The male voice speaks first, followed by the female voice",
    "B": "The female voice speaks first, followed by the male voice",
    "C": "Both voices speak simultaneously",
    "D": "Only one person speaks in this segment"
  },
  "correct_answer": "A",
  "task_type": "task1"
}
```

---

### Quality Control

#### `general_check.py`
**Purpose**: Perform general quality checks on generated questions.

**What It Does**:
- Validates question format and completeness
- Checks answer consistency
- Ensures all required fields are present
- Identifies formatting issues

**Checks Performed**:
- Question has proper structure
- All 4 options (A/B/C/D) are present
- Correct answer is specified
- No duplicate options
- Reasonable question length
- Proper grammar and spelling

**Usage**:
```bash
python general_check.py \
    --data_path ./final_qa.json \
    --output_path ./check_report.json
```

**Output**: Report of issues found with suggestions for fixes

---

#### `specific_check.py`
**Purpose**: Perform task-specific quality checks.

**What It Does**:
- Validates questions match their task type requirements
- Ensures correct cognitive dimension and audio type
- Checks multi-modal requirements are met
- Verifies scene span consistency

**Task-Specific Checks**:
- **Temporal tasks** (task1, 8, 10, 11): Question mentions time/order
- **Spatial tasks** (task2, 4, 5): Question mentions location/position
- **Emotional tasks** (task6, 7, 16): Question mentions emotion/mood
- **Plot tasks** (task12, 13, 17): Question mentions narrative/story
- **Long-form tasks** (task9, 15): Question requires extended context

**Usage**:
```bash
python specific_check.py \
    --data_path ./final_qa.json \
    --output_path ./specific_check_report.json
```

**Validation Rules**:
- Question requires both audio AND visual information
- Question cannot be answered with single modality
- Difficulty appropriate for task type
- Answer is unambiguous

---

### Supporting Files

#### `parse_data.py`
**Purpose**: Parse and preprocess caption data for question generation.

**Key Functions**:
- `load_captions()`: Load audio and visual captions
- `align_captions()`: Align audio and visual by timestamp
- `extract_segment()`: Extract caption content for specific time range
- `merge_scenes()`: Combine captions across scene boundaries
- `clean_captions()`: Remove noise and normalize text

**Usage**: Imported by other scripts for data processing

---

#### `clean_data.py`
**Purpose**: Clean and normalize generated questions.

**What It Does**:
- Removes formatting artifacts
- Normalizes text (punctuation, capitalization)
- Fixes common generation errors
- Standardizes option labels

**Cleaning Operations**:
- Remove extra whitespace
- Fix punctuation
- Normalize quote marks
- Remove duplicate sentences
- Fix option formatting

**Usage**:
```bash
python clean_data.py \
    --input_path ./raw_qa.json \
    --output_path ./cleaned_qa.json
```

---

#### `gather_qa.py`
**Purpose**: Gather and organize questions from multiple sources/batches.

**What It Does**:
- Combines QA pairs from multiple generation runs
- Removes duplicates
- Organizes by task type
- Creates final benchmark file

**Usage**:
```bash
python gather_qa.py \
    --input_dir ./qa_batches \
    --output_path ./benchmark.json
```

---

#### `prompts.py`
**Purpose**: Stores all prompt templates used in generation.

**Contents**:
- System prompts for LLM
- Task-specific question generation prompts
- Distractor generation prompts
- Validation prompts

**Prompt Categories**:
1. **Interval Identification Prompts**: Guide LLM to find suitable segments
2. **Question Generation Prompts**: Task-specific templates (task1-17)
3. **Distractor Generation Prompts**: Create plausible wrong answers
4. **Validation Prompts**: Check question quality

**Example Prompt Structure**:
```python
TASK1_PROMPT = """
Based on the following audio-visual captions, generate a question that requires
understanding the TEMPORAL ORDER of speeches.

Visual Captions: {visual_captions}
Audio Captions: {audio_captions}

Requirements:
- Question must require both audio (speech content) and visual (speaker identity) information
- Focus on temporal ordering
- Generate 4-5 word question
"""
```

---

#### `utils.py`
**Purpose**: Shared utility functions for the generation pipeline.

**Key Functions**:
- `load_json()`, `save_json()`: JSON I/O
- `timestamp2seconds()`: Time format conversion
- `task2audform`: Mapping of tasks to audio types
- `merge_captions_segment_wise()`: Combine captions
- `clean_subtitles()`: Clean subtitle text
- `load_caption()`: Load and parse captions

**Mappings**:
```python
task2audform = {
    "task1": ['speech'],
    "task2": ['speech_emotion'],
    "task4": ['sound_event'],
    "task6": ['speech_emotion'],
    "task7": ['music'],
    ...
}
```

---

## Generation Pipeline Flow

```
1. Input: Audio-Visual Captions
         ↓
   ┌─────────────────────────────┐
   │  identify_intervals.py      │
   │  Find suitable segments     │
   └─────────────────────────────┘
         ↓
   Intervals with task assignments
         ↓
   ┌─────────────────────────────┐
   │  generate_qa.py             │
   │  Create questions           │
   └─────────────────────────────┘
         ↓
   Question-Answer pairs
         ↓
   ┌─────────────────────────────┐
   │  generate_distractor.py     │
   │  Add wrong options          │
   └─────────────────────────────┘
         ↓
   Multiple-choice questions
         ↓
   ┌─────────────────────────────┐
   │  Quality Control            │
   │  - general_check.py         │
   │  - specific_check.py        │
   └─────────────────────────────┘
         ↓
   ┌─────────────────────────────┐
   │  clean_data.py              │
   │  Clean and normalize        │
   └─────────────────────────────┘
         ↓
   ┌─────────────────────────────┐
   │  gather_qa.py               │
   │  Organize into benchmark    │
   └─────────────────────────────┘
         ↓
2. Output: Final Benchmark
```

## Task Types and Requirements

| Task | Cognitive | Audio Type | Scene | Requirements |
|------|-----------|------------|-------|--------------|
| task1 | Temporal | Speech | Single | Speech order in time |
| task2 | Spatial | Speech+Emotion | Single | Speaker location + emotion |
| task4 | Spatial | Sound Event | Single | Sound source location |
| task5 | Spatial | Sound Event | Single | Sound-object association |
| task6 | Emotional | Speech+Emotion | Single | Emotion recognition |
| task7 | Emotional | Music | Single | Music mood |
| task8 | Temporal | Speech | Single | Speech timing |
| task9 | Long-form | Speech | Multi | Cross-scene speech |
| task10 | Temporal | Speech+Emotion | Multi | Multi-scene temporal |
| task11 | Temporal | Speech | Multi | Dialogue tracking |
| task12 | Plot | Sound+Music | Multi | Narrative |
| task13 | Plot | Speech | Multi | Story |
| task15 | Long-form | All | Full | Full video |
| task16 | Emotional | Music | Full | Long music |
| task17 | Plot | Speech | Full | Complete narrative |

## Quality Assurance Principles

### 1. Multi-modal Requirement
Every question must require BOTH audio and visual information:
- ✅ "What emotion does the speaker show while saying 'I'm fine'?" (needs audio + visual)
- ❌ "What does the person say?" (audio only)
- ❌ "What is the person wearing?" (visual only)

### 2. Grounded in Evidence
Questions and answers must be based on caption content:
- No speculation beyond provided information
- Answers verifiable from captions
- No ambiguous or subjective questions

### 3. Plausible Distractors
Wrong options must be believable:
- Not obviously incorrect
- Contextually relevant
- Different enough from correct answer
- Similar difficulty level

### 4. Cognitive Diversity
Questions span multiple reasoning types:
- Temporal (time-based)
- Spatial (location-based)
- Emotional (mood/feeling)
- Long-form (extended context)
- Plot (narrative understanding)

## Models Used

### Question Generation
- **Model**: Qwen2.5-72B-Instruct or GPT-4
- **Purpose**: Generate questions from captions
- **Input**: Audio-visual captions + prompt template
- **Output**: Question + answer

### Distractor Generation
- **Model**: Same as question generation
- **Purpose**: Create plausible wrong options
- **Input**: Question + correct answer + captions
- **Output**: 3 distractor options

### Quality Checking
- **Model**: Same LLM
- **Purpose**: Validate question quality
- **Input**: Question + metadata
- **Output**: Quality score + feedback

## Configuration

Key parameters in scripts:

```python
# Model selection
MODEL_PATH = "Qwen/Qwen2.5-72B-Instruct"  # or path to local model

# API configuration
API_KEY = os.environ.get('QWEN_API_KEY', '')  # Load from environment

# Generation parameters
BATCH_SIZE = 48
TEMPERATURE = 0.7
MAX_TOKENS = 2048
```

## Dependencies

- `openai`: OpenAI API client (for GPT models)
- `transformers`: Hugging Face models
- `vllm`: Fast LLM inference
- `torch`: PyTorch
- `tqdm`: Progress bars

## Best Practices

### For Interval Identification
- Use clear task requirements
- Consider audio-visual content balance
- Validate sufficient information exists

### For Question Generation
- Use task-specific prompts
- Ensure multi-modal requirements
- Validate against captions
- Generate diverse question types

### For Distractor Generation
- Make distractors plausible
- Avoid obvious wrong answers
- Ensure clear difference from correct answer
- Test with humans if possible

### For Quality Control
- Run both general and specific checks
- Fix issues before finalizing
- Manually review sample questions
- Iterate based on evaluation results

## Notes

- Generation is largely automated but benefits from human review
- Prompt engineering is critical for quality
- Different LLMs may require prompt adjustments
- Quality checking catches most issues but not all
- Final benchmark should be manually validated on a sample
