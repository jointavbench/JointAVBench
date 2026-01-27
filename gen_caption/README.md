# Caption Generation Module

This module generates detailed audio and visual captions from raw video files. Captions serve as the foundation for question generation by providing rich, timestamped descriptions of video content.

## Overview

The caption generation process extracts both audio and visual information from videos:
- **Audio Captions**: Speech transcription, emotion recognition, sound events, music analysis
- **Video Captions**: Scene descriptions, character actions, visual dynamics, emotional tone

## Files

### Core Caption Generation

#### `audio_caption.py`
**Purpose**: Generate comprehensive audio captions including speech, emotion, and audio events.

**Key Features**:
- Uses Qwen2.5-Omni-7B model for audio analysis
- Analyzes three types of audio information:
  - **Speech Content**: Transcribes spoken words and dialogue
  - **Speech Emotion**: Detects emotional tone (happy, sad, angry, neutral, etc.)
  - **Audio Events**: Identifies sound events (footsteps, door closing, music, etc.)
- Processes audio in segments aligned with video scenes
- Outputs timestamped captions with start/end times

**Usage**:
```bash
python audio_caption.py \
    --task-name audio_caption \
    --data-dir /path/to/caption/data \
    --audio-dir /path/to/audio/files
```

**Task Types**:
- `audio_caption`: General audio analysis (sound events, music)
- `speech_emotion`: Speech emotion recognition
- `test`: Testing mode

**Output Format**:
```json
{
  "video_name": "example_video",
  "type": "audio_caption",
  "segment_id": 1,
  "start_time": "00:00:05.000",
  "end_time": "00:00:10.000",
  "caption": "Male voice speaking with neutral emotion, background music playing"
}
```

---

#### `audio_caption_vllm.py`
**Purpose**: Alternative audio caption generation using vLLM for faster inference.

**Differences from `audio_caption.py`**:
- Uses vLLM backend for accelerated inference
- Better for batch processing multiple videos
- Same model (Qwen2.5-Omni-7B) but optimized serving

**Usage**: Same as `audio_caption.py`

---

#### `video_caption.py`
**Purpose**: Generate detailed visual captions describing video content.

**Key Features**:
- Uses vision-language models (vLLM) for video understanding
- Performs scene detection to segment videos
- Analyzes each scene for:
  - **Scene Setting**: Environment, location, lighting
  - **Characters & Actions**: Appearance, movements, interactions
  - **Scene Dynamics**: Events, pacing, changes over time
  - **Emotional Tone**: Mood and atmosphere from visual cues
  - **Key Events**: Important narrative moments
- Generates dense temporal annotations

**Usage**:
```bash
python video_caption.py \
    --input_video /path/to/video.mp4 \
    --output_dir /path/to/output
```

**Output Format**:
```json
{
  "video_name": "example_video",
  "segment_id": 1,
  "start_time": "00:00:05.000",
  "end_time": "00:00:10.000",
  "caption": "A person walks into a dimly lit room. They approach a desk and pick up a book...",
  "scene_type": "indoor",
  "key_objects": ["desk", "book", "person"]
}
```

---

### Utility Files

#### `utils.py`
**Purpose**: Shared utility functions for caption generation.

**Key Functions**:
- `PySceneSegmenter`: Scene boundary detection using PySceneDetect
- `timestamp2seconds()`: Convert timestamp strings to seconds
- `seconds2timestamp()`: Convert seconds to timestamp format
- `load_json()`, `save_json()`: JSON file I/O
- `load_jsonl()`: Load JSONL files

**Scene Detection**:
Uses content-based detection to identify scene changes, which helps:
- Break videos into manageable segments
- Ensure captions align with scene boundaries
- Provide temporal structure for question generation

---

#### `vision_process.py`
**Purpose**: Vision preprocessing utilities for video caption generation.

**Key Functions**:
- Frame extraction from videos
- Image preprocessing and normalization
- Batch processing of video frames
- Integration with vision-language models
- Frame sampling strategies (uniform, keyframe-based)

**Features**:
- Handles various video formats (MP4, AVI, MOV)
- Optimized frame extraction with decord
- Memory-efficient processing for long videos
- GPU acceleration support

---

## Caption Generation Pipeline

```
Input: Raw Video File
         ↓
    ┌────────────────────────┐
    │  Scene Detection       │
    │  (PySceneSegmenter)    │
    └────────────────────────┘
         ↓
    ┌────────────────────────┐
    │  Audio Extraction      │
    │  (segment audio)       │
    └────────────────────────┘
         ↓
    ┌────────────────────────┐
    │  Visual Captioning     │
    │  (video_caption.py)    │
    └────────────────────────┘
         ↓
    Scene descriptions with timestamps
         ↓
    ┌────────────────────────┐
    │  Audio Captioning      │
    │  (audio_caption.py)    │
    └────────────────────────┘
         ↓
    Speech, emotion, sound event annotations
         ↓
Output: Timestamped Audio-Visual Captions
```

## Caption Types

### Audio Captions

#### 1. Speech Content
- Transcribes spoken dialogue
- Identifies speakers
- Captures conversation flow

**Example**:
```
"Male voice: 'Where are you going?'"
"Female voice responds: 'To the store'"
```

#### 2. Speech Emotion
- Analyzes vocal tone, pitch, pace
- Detects emotions: happy, sad, angry, fearful, surprised, disgusted, neutral
- Separates vocal emotion from content meaning

**Example**:
```
"Speaker sounds anxious, with elevated pitch and rapid speech"
```

#### 3. Audio Events
- Sound effects: footsteps, door closing, glass breaking
- Music: background score, songs
- Environmental sounds: rain, traffic, crowds

**Example**:
```
"Door slams shut, followed by hurried footsteps"
"Suspenseful music begins playing"
```

### Visual Captions

Describe scene composition, character actions, and visual dynamics:

**Example**:
```
"In a dimly lit office, a man in a suit sits at a desk reviewing documents.
He looks up with a concerned expression as someone enters through the door behind him.
The room has bookshelves lining the walls and a window showing the city at night."
```

## Models Used

### Audio Captioning
- **Primary Model**: Qwen2.5-Omni-7B
- **Capabilities**: Speech recognition, emotion detection, sound event recognition
- **Input**: Audio segments (5-30 seconds)
- **Output**: Text descriptions with timestamps

### Video Captioning
- **Primary Model**: Vision-Language Models via vLLM
- **Capabilities**: Scene understanding, action recognition, visual reasoning
- **Input**: Video frames sampled from scenes
- **Output**: Detailed scene descriptions

## Data Flow

1. **Input**: Raw video file (MP4, AVI, etc.)
2. **Scene Detection**: Split video into scenes
3. **Parallel Processing**:
   - Extract audio → Generate audio captions
   - Extract frames → Generate video captions
4. **Alignment**: Merge captions by timestamp
5. **Output**: Combined audio-visual captions with precise timing

## Output Format

Captions are saved in JSON format with the following structure:

```json
{
  "video_name": "example_video_001",
  "duration": "00:02:30.000",
  "scenes": [
    {
      "scene_id": 1,
      "start_time": "00:00:00.000",
      "end_time": "00:00:15.000",
      "visual_caption": "A person enters a room...",
      "audio_captions": [
        {
          "type": "speech",
          "content": "Hello, is anyone here?",
          "emotion": "curious",
          "speaker": "male"
        },
        {
          "type": "sound_event",
          "content": "Door creaking open"
        }
      ]
    }
  ]
}
```

## Key Design Principles

### 1. Objective Description
- Only describe what is explicitly visible/audible
- No speculation or inference beyond evidence
- Separate observations from interpretations

### 2. Temporal Precision
- All captions include exact timestamps
- Aligned with scene boundaries
- Support both short and long videos

### 3. Multi-modal Coverage
- Visual information (what you see)
- Audio information (what you hear)
- Combined for complete understanding

### 4. Granularity Control
- Scene-level captions for context
- Segment-level for specific events
- Frame-level for detailed analysis

## Usage in Benchmark Generation

These captions feed into the question generation pipeline:

1. **Interval Identification**: Find suitable segments for each task type
2. **Question Generation**: Create questions based on caption content
3. **Answer Verification**: Ensure answers are grounded in captions
4. **Multi-modal Requirements**: Questions need both audio and visual captions

## Dependencies

- `transformers`: Hugging Face models
- `torch`: PyTorch for model inference
- `vllm`: Fast LLM inference
- `decord`: Video decoding
- `scenedetect`: Scene boundary detection
- `librosa`: Audio processing
- `opencv-python`: Video processing

## Notes

- Captions are the foundation of the benchmark - quality here determines question quality
- Dense temporal annotations enable complex multi-scene questions
- Separate audio and visual processing allows modality-specific question types
- Models can be swapped for different capabilities or languages
