# Technical Report: Storyline-based Video Creation with Vidi

## Executive Summary

This technical report analyzes the **Storyline-based Video Creation** capability mentioned in the Vidi2 paper ([arXiv:2511.19529](https://arxiv.org/abs/2511.19529)). While the current open-source release focuses on temporal retrieval (VUE-TR) and spatio-temporal grounding (VUE-STG), we provide insights into how the Vidi architecture could support storyline-based video creation and smart video clipping.

## 1. Introduction

### 1.1 What is Vidi?

Vidi is a family of Large Multimodal Models (LMMs) developed by ByteDance for video understanding and editing (VUE) scenarios. The project represents a significant advancement in AI-powered video processing, combining:

- **Visual Understanding**: Frame-by-frame video analysis using CLIP/SigLIP encoders
- **Audio Processing**: Whisper-based audio feature extraction
- **Temporal Reasoning**: Multi-modal attention mechanisms for temporal localization
- **Natural Language Interface**: Text-based querying for video content

### 1.2 Current Release Scope

The public repository includes:
- **VUE-TR (Temporal Retrieval)**: Finding specific moments in videos based on text queries
- **VUE-STG (Spatio-Temporal Grounding)**: Locating objects/events with both temporal and spatial precision
- **Vidi-7B Model**: A 7-billion parameter model for inference

## 2. Understanding Storyline-based Video Creation

### 2.1 Concept Overview

According to the Vidi2 paper, Storyline-based Video Creation involves:

1. **Content Analysis**: Deep understanding of video content, including scenes, actions, emotions, and narrative elements
2. **Storyline Generation**: Automatic or guided creation of narrative structures
3. **Smart Clipping**: Intelligent selection and arrangement of video segments
4. **Coherent Assembly**: Creating cohesive video outputs that tell a story

### 2.2 Relationship to Current Features

The implemented features provide foundational capabilities for storyline-based creation:

| Feature | Role in Storyline Creation |
|---------|---------------------------|
| Temporal Retrieval | Locating relevant scenes for the storyline |
| Highlight Detection | Identifying key moments to include |
| Spatio-Temporal Grounding | Tracking subjects across scenes |
| VQA | Understanding context and relationships |

## 3. Technical Architecture

### 3.1 Model Architecture

The Vidi-7B model uses a sophisticated multi-modal architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Vidi Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌───────────┐    ┌───────────┐    ┌───────────────────────┐  │
│   │  Video    │    │  Audio    │    │  Text Query           │  │
│   │  Frames   │    │  Waveform │    │  (e.g., storyline)    │  │
│   └─────┬─────┘    └─────┬─────┘    └───────────┬───────────┘  │
│         │                │                      │               │
│         ▼                ▼                      ▼               │
│   ┌───────────┐    ┌───────────┐    ┌───────────────────────┐  │
│   │ CLIP/     │    │ Whisper   │    │ Mistral Tokenizer     │  │
│   │ SigLIP    │    │ Encoder   │    │                       │  │
│   │ Encoder   │    │           │    │                       │  │
│   └─────┬─────┘    └─────┬─────┘    └───────────┬───────────┘  │
│         │                │                      │               │
│         ▼                ▼                      ▼               │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │            Multi-Modal Fusion Layer                      │  │
│   │   (Diagonal Attention for V2V, A2A, T2T, T2V, T2A)      │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              Mistral Decoder Layers (32)                 │  │
│   │           with Cross-Attention Mechanisms                │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                  Output (Time Ranges)                    │  │
│   │           e.g., "0.15-0.25, 0.45-0.60"                  │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Components

#### 3.2.1 Vision Encoder
- Based on CLIP-ViT-Large-Patch14 or SigLIP
- Processes video frames at 1 FPS (configurable)
- Supports dynamic resolution with aspect ratio preservation
- Grid-based image splitting for high-resolution processing

#### 3.2.2 Audio Encoder
- Based on Whisper-Large-V3
- 16kHz sampling rate
- Processes audio in 30-second chunks
- Extracts rich audio features including speech and non-speech sounds

#### 3.2.3 Multi-Modal Fusion
The Diagonal Attention (Dattn) mechanism enables:
- **T2T**: Text self-attention for language modeling
- **T2V**: Text-to-Vision cross-attention for visual grounding
- **T2A**: Text-to-Audio cross-attention for audio grounding
- **V2V**: Visual self-attention for temporal coherence
- **A2A**: Audio self-attention for acoustic coherence

### 3.3 Inference Pipeline

```python
# Simplified inference flow
def process_video_for_storyline(video_path, storyline_query, model):
    # 1. Load and preprocess video
    video_frames = load_video(video_path, fps=1)
    video_tensor = process_images(video_frames, image_processor)
    
    # 2. Load and preprocess audio
    audio = load_audio(video_path)
    audio_tensor = process_audio(audio, audio_processor)
    
    # 3. Prepare query for storyline
    query = f"Find segments that match this storyline: {storyline_query}"
    prompt = create_prompt(query, video_length)
    
    # 4. Generate temporal predictions
    output = model.generate(
        input_ids=prompt,
        images=video_tensor,
        audios=audio_tensor,
        max_new_tokens=1024
    )
    
    # 5. Parse time ranges
    time_ranges = parse_output(output)
    
    return time_ranges
```

## 4. Storyline-based Video Creation Workflow

### 4.1 Proposed Workflow

Based on the architecture, here's how storyline-based video creation could work:

```
┌────────────────────────────────────────────────────────────────┐
│               Storyline-based Video Creation                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT:                                                         │
│  ┌─────────────┐  ┌────────────────────────────────────────┐  │
│  │ Source Video │  │ Storyline Description                   │  │
│  │             │  │ "Opening shot → Character intro →       │  │
│  │             │  │  Conflict → Resolution → Ending"        │  │
│  └─────────────┘  └────────────────────────────────────────┘  │
│                                                                 │
│  PROCESS:                                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ 1. Parse Storyline into Segments                         │  │
│  │    └─→ ["Opening shot", "Character intro", ...]          │  │
│  │                                                           │  │
│  │ 2. For each segment, use Vidi to find matching clips     │  │
│  │    └─→ Temporal Retrieval + Scoring                      │  │
│  │                                                           │  │
│  │ 3. Rank and select best clips                            │  │
│  │    └─→ Consider relevance, diversity, transitions        │  │
│  │                                                           │  │
│  │ 4. Assemble clips in storyline order                     │  │
│  │    └─→ Apply transitions, adjust pacing                  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  OUTPUT:                                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Final edited video following the storyline               │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 Example Use Cases

1. **Event Highlight Reels**
   - Input: 2-hour wedding video
   - Storyline: "Ceremony highlights → Vows exchange → Ring exchange → First kiss → Reception entrance → First dance → Cake cutting → Toasts"
   - Output: 10-minute curated highlight video

2. **Sports Recap**
   - Input: Full match recording
   - Storyline: "Opening kickoff → First scoring attempt → Goals → Key saves → Yellow/Red cards → Final whistle → Celebrations"
   - Output: 5-minute match summary

3. **Documentary Creation**
   - Input: Hours of interview footage
   - Storyline: "Introduction → Problem statement → Expert opinions → Solutions → Conclusion"
   - Output: Structured documentary

## 5. Implementation Guide

### 5.1 Environment Setup

```bash
# Option 1: Using the conda setup script
./setup_conda.sh

# Option 2: Manual setup with conda
conda env create -f environment.yml
conda activate vidi

# Install flash-attention for GPU acceleration (optional but recommended)
pip install flash-attn==2.6.3 --no-build-isolation
```

### 5.2 Running Inference

```bash
# Basic temporal retrieval
cd Vidi_7B
python3 inference.py \
    --video-path /path/to/video.mp4 \
    --query "person giving a presentation" \
    --model-path /path/to/vidi-7b-model
```

### 5.3 Extending for Storyline-based Creation

```python
# Example extension for storyline-based clipping
import re
from model.builder import load_pretrained_model
from model.vid_utils import load_video, load_audio, process_audio
from model.img_utils import process_images
from model.txt_utils import tokenizer_image_token, preprocess_chat

def storyline_clip(video_path, storyline_segments, model_path):
    """
    Create video clips based on a storyline.
    
    Args:
        video_path: Path to source video
        storyline_segments: List of segment descriptions
        model_path: Path to Vidi model
        
    Returns:
        List of (start_time, end_time, segment_name) tuples
    """
    # Load model
    model, tokenizer, image_processor, audio_processor = load_pretrained_model(model_path)
    
    # Load video and audio
    video = load_video(video_path)
    video = process_images(video, image_processor, model.config)
    audio = load_audio(video_path, audio_processor.sampling_rate)
    audio_tensor, audio_size = process_audio(audio, audio_processor)
    
    # Get video length
    video_length = get_video_length(video_path)
    
    clips = []
    for segment in storyline_segments:
        # Query model for this segment
        result = query_segment(
            segment, 
            video, 
            audio_tensor, 
            audio_size,
            video_length,
            model, 
            tokenizer
        )
        clips.append((result['start'], result['end'], segment))
    
    return clips
```

## 6. Technical Specifications

### 6.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA RTX 3090 (24GB) | NVIDIA A100 (40GB) |
| RAM | 32GB | 64GB |
| Storage | 50GB (model + dependencies) | 100GB+ |
| CUDA | 11.7+ | 12.1+ |

### 6.2 Software Dependencies

- **Python**: 3.10
- **PyTorch**: 2.3.1 with CUDA 12.1
- **Transformers**: 4.44.2
- **Flash-Attention**: 2.6.3 (recommended)
- **FFmpeg**: For video/audio processing

### 6.3 Model Specifications

| Model | Parameters | Context Length | Modalities |
|-------|------------|----------------|------------|
| Vidi-7B | 7B | 4096 tokens | Video + Audio + Text |
| Vidi2 | Not released | - | Enhanced STG |

## 7. Evaluation Metrics

### 7.1 Temporal Retrieval (VUE-TR)

- **IoU (Intersection over Union)**: Overlap between predicted and ground-truth time ranges
- **Precision**: Accuracy of predicted time ranges
- **Recall**: Coverage of ground-truth time ranges

### 7.2 Spatio-Temporal Grounding (VUE-STG)

- **t_IoU**: Temporal intersection over union
- **v_IoU**: Volume-based IoU (3D)
- **t_Precision / t_Recall**: Temporal precision and recall
- **v_Precision / v_Recall**: Volumetric precision and recall

### 7.3 Vidi2 Results (as of Nov 2025)

| Model | t_Precision | t_Recall | t_IoU | v_Precision | v_Recall | v_IoU |
|-------|-------------|----------|-------|-------------|----------|-------|
| **Vidi2** | **0.730** | **0.598** | **0.532** | **0.446** | **0.363** | **0.326** |
| Gemini-3-Pro | 0.519 | 0.353 | 0.275 | 0.090 | 0.057 | 0.046 |
| GPT-5 | 0.383 | 0.195 | 0.164 | 0.130 | 0.065 | 0.055 |

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **Storyline Creation Not Yet Open-Sourced**: The full storyline-based video creation pipeline mentioned in the paper is not yet available in the public release
2. **GPU Requirements**: The 7B model requires significant GPU memory
3. **Processing Speed**: Long videos require extended processing time
4. **Language**: Best performance with English queries

### 8.2 Future Directions

1. **Full Storyline Pipeline**: Expected release of storyline-based creation features
2. **Smaller Models**: Distilled versions for edge deployment
3. **Real-time Processing**: Optimizations for live video processing
4. **Multi-language Support**: Extended language capabilities

## 9. Conclusion

The Vidi project represents a significant advancement in AI-powered video understanding. While the current open-source release focuses on temporal retrieval and spatio-temporal grounding, these capabilities provide the foundation for storyline-based video creation. The multi-modal architecture combining video, audio, and text understanding enables intelligent video clipping that goes beyond simple scene detection.

For users interested in storyline-based video creation, the current temporal retrieval features can be leveraged as building blocks, with the expectation that more comprehensive storyline generation capabilities will be released in future updates.

## 10. References

1. Vidi2 Paper: [arXiv:2511.19529](https://arxiv.org/abs/2511.19529)
2. Vidi1 Paper: [arXiv:2504.15681](https://arxiv.org/abs/2504.15681)
3. Project Homepage: [https://bytedance.github.io/vidi-website/](https://bytedance.github.io/vidi-website/)
4. Demo: [https://vidi.byteintl.com/](https://vidi.byteintl.com/)
5. Model Weights: [https://huggingface.co/bytedance-research/Vidi-7B](https://huggingface.co/bytedance-research/Vidi-7B)

---

*Report generated for the Vidi project repository*
*Last updated: December 2025*
