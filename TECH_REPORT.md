# Technical Report: Storyline-based Video Creation with Vidi2.5

> **Paper**: [Vidi2.5: Large Multimodal Models for Video Understanding and Creation](https://arxiv.org/abs/2511.19529)
> **Authors**: Intelligent Editing Team, Intelligent Creation, ByteDance Inc.
> **Repository**: <https://github.com/bytedance/vidi>
> **Demo**: <https://vidi.byteintl.com/>

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background and Motivation](#2-background-and-motivation)
3. [Model Architecture](#3-model-architecture)
4. [Storyline-based Video Creation — Vidi-Edit](#4-storyline-based-video-creation--vidi-edit)
   - 4.1 [Problem Definition](#41-problem-definition)
   - 4.2 [Editing Plan Specification](#42-editing-plan-specification)
   - 4.3 [Post-Training for Editing Plan Generation](#43-post-training-for-editing-plan-generation)
   - 4.4 [Execution Pipeline](#44-execution-pipeline)
5. [Supporting Capabilities in This Repository](#5-supporting-capabilities-in-this-repository)
   - 5.1 [Temporal Retrieval](#51-temporal-retrieval)
   - 5.2 [Spatio-Temporal Grounding](#52-spatio-temporal-grounding)
   - 5.3 [Plot Understanding and Reasoning](#53-plot-understanding-and-reasoning)
6. [Environment Setup Guide](#6-environment-setup-guide)
7. [Benchmark and Evaluation](#7-benchmark-and-evaluation)
8. [Current Limitations and Availability](#8-current-limitations-and-availability)
9. [Conclusion](#9-conclusion)
10. [References](#10-references)

---

## 1. Executive Summary

Vidi2.5 is a family of Large Multimodal Models (LMMs) developed by ByteDance for comprehensive video understanding and creation. The flagship application is **Vidi-Edit**, a post-trained variant capable of generating structured **editing plans** that transform raw video assets into polished, narrative-driven videos — complete with storyline, voiceover, music, and visual effects.

This report focuses specifically on the **Storyline-based Video Creation** capability (referred to as "Vidi-Edit" in the paper), which is the most advanced and application-relevant feature of the Vidi2.5 system. Unlike simple highlight extraction or video trimming, Vidi-Edit reasons about the narrative structure across multiple video clips and produces a coherent editing plan that downstream systems can execute.

### Key Takeaways

| Aspect | Detail |
|--------|--------|
| **What it does** | Generates structured editing plans from raw video assets |
| **Input** | Multiple raw video clips + optional user intent (text prompt) |
| **Output** | Editing plan specifying narrative structure, voiceover, audio attributes, and visual effects |
| **Underlying model** | Vidi2.5 (12B parameters), post-trained as Vidi-Edit |
| **Architecture** | Multimodal LMM with text + vision (SigLIP + CLIP) + audio (Whisper) encoders |
| **Open-source code** | Inference/finetune code for base models (Vidi-7B, Vidi1.5-9B) is released; Vidi-Edit's post-training data and specific pipeline code are not yet publicly available |

---

## 2. Background and Motivation

Video has become the dominant medium for online communication, yet high-quality video production remains challenging for most users, particularly on mobile devices. Key pain points include:

- **Clip selection**: Choosing the most relevant segments from hours of raw footage
- **Narrative structure**: Arranging clips into a coherent storyline (not just concatenation)
- **Audio design**: Selecting appropriate background music, adding voiceover
- **Visual effects**: Applying transitions, emphasis cues, and stylistic elements
- **Temporal alignment**: Synchronizing all elements with precise timing

Traditional video editing tools (e.g., Adobe Premiere, Final Cut Pro) require significant manual effort and expertise. The Vidi project aims to automate this entire workflow using a Large Multimodal Model that understands video content at a deep semantic level.

The evolution of the Vidi models reflects this goal:

| Version | Release | Focus |
|---------|---------|-------|
| Vidi (7B) | Apr 2025 | Temporal retrieval (TR) — finding time ranges matching text queries |
| Vidi1.5 (9B) | Aug 2025 | Enhanced TR + basic video QA |
| Vidi2 (12B) | Nov 2025 | Spatio-temporal grounding (STG) + video QA + highlight/chapter extraction |
| Vidi2.5 | Jan 2026 | RL-enhanced STG/TR + thinking model + **Vidi-Edit for storyline-based video creation** |

---

## 3. Model Architecture

Vidi2.5 retains a multimodal architecture that jointly processes text, visual, and audio inputs:

```
┌─────────────────────────────────────────────────────┐
│                    Vidi2.5 (12B)                     │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Vision   │  │  Audio   │  │  Text (Tokenizer)│  │
│  │  Encoder  │  │  Encoder │  │                  │  │
│  │ SigLIP2 + │  │ Whisper  │  │  Sentencepiece   │  │
│  │  CLIP     │  │ Large V3 │  │                  │  │
│  └─────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
│        │              │                 │            │
│        ▼              ▼                 ▼            │
│  ┌──────────────────────────────────────────────┐   │
│  │          Multimodal Projection Layers         │   │
│  │     (Learned MLP + Cross-Attention Fusion)    │   │
│  └──────────────────┬───────────────────────────┘   │
│                     ▼                               │
│  ┌──────────────────────────────────────────────┐   │
│  │           LLM Backbone (Gemma-3)              │   │
│  │    with Dense Attention (DATTN) + Flash-Attn  │   │
│  │    + Sequence Parallelism                     │   │
│  └──────────────────┬───────────────────────────┘   │
│                     ▼                               │
│              Structured Output                      │
│   (timestamps, bounding boxes, text, plans, etc.)   │
└─────────────────────────────────────────────────────┘
```

### Key Design Choices

- **Dual vision encoders**: SigLIP2-SO400M for semantic features + CLIP for complementary visual representations
- **Adaptive token compression**: Dynamically adjusts visual token count based on video length, balancing short and long video representation efficiency
- **Unified image/video encoding**: A single image is treated as a 1-second silent video
- **Audio-visual alignment**: Whisper-Large-V3 processes the audio track in parallel with the visual stream
- **RL training** (Vidi2.5): Reinforcement learning with verifiable rewards for consistent improvement on STG, TR, and Video QA

---

## 4. Storyline-based Video Creation — Vidi-Edit

### 4.1 Problem Definition

**Video Editing Planning** is defined as follows:

> **Given**: A collection of raw assets (images and/or videos) and optional user prompts (editing requirements)
>
> **Generate**: An editing plan that specifies the narrative structure, narration content, audio attributes, and visual editing intent

This is fundamentally different from traditional video understanding tasks:

| Task | Input | Output |
|------|-------|--------|
| Temporal Retrieval | Video + query | Time ranges |
| Spatio-Temporal Grounding | Video + query | Time ranges + bounding boxes |
| Video QA | Video + question | Text answer |
| **Video Editing Planning** | **Multiple videos + optional intent** | **Structured editing plan** |

The editing plan is a **high-level textual/structured representation** that unifies narrative structure, audio design, and visual intent. It can be consumed by downstream execution systems to produce the final rendered video.

### 4.2 Editing Plan Specification

The editing plan consists of four interdependent components:

#### 4.2.1 Narrative Structure

Defines the hierarchical and temporal orchestration of the video:

- **Clip selection**: Which raw clips to include and which to omit
- **Segment extraction**: Which portions of selected clips to use (start/end times)
- **Segment ordering**: How to arrange the selected segments for narrative flow

This establishes the temporal and semantic backbone of the edit. The model determines an optimal ordering that conveys the intended narrative — including context setup (opening hook), content development, and a concluding wrap-up.

#### 4.2.2 Voiceover Content

Specifies the narrated material aligned with the narrative structure:

- **Semantic content**: What the narrator says
- **Delivery style**: Tone, pacing, emotion
- **Temporal alignment**: When each narration segment plays relative to the video

#### 4.2.3 Audio Attributes

Characterizes auditory elements that complement the visual narrative:

- **Music style**: Genre, mood
- **Rhythmic profile**: Tempo, beat pattern
- **Speaker attributes**: Voice characteristics for TTS

#### 4.2.4 Visual Editing Intent

Articulates stylistic and rhythmic presentation directives:

- **Transition design**: How clips connect (cuts, fades, wipes, etc.)
- **Emphasis cues**: Zoom, slow-motion, text overlays
- **Stylistic directives**: Color grading, filter effects

Together, these four components form a **complete semantic blueprint** for video creation.

### 4.3 Post-Training for Editing Plan Generation

Vidi-Edit is created by **post-training** the base Vidi2.5 model:

1. **No architecture changes**: The same 12B multimodal architecture is used
2. **Task-specific supervision**: The model is fine-tuned on multimodal inputs paired with textual/structured editing plans
3. **Format adherence**: Training emphasizes conformance to the planning format specification
4. **Semantic consistency**: The supervision signal enforces consistency across all four plan components (narrative, voiceover, audio, visual intent)

This approach leverages the strong spatio-temporal understanding already present in Vidi2.5 and directs it toward the structured output format required for editing plans.

### 4.4 Execution Pipeline

The Vidi-Edit execution pipeline separates **planning** from **execution**:

```
Raw Video Assets + User Intent
           │
           ▼
    ┌──────────────┐
    │   Vidi-Edit   │  ← Planning (the model)
    │  (Vidi2.5     │
    │  post-trained)│
    └──────┬───────┘
           │
           ▼
    Structured Editing Plan
    (narrative + voiceover + audio + visual intent)
           │
           ▼
    ┌──────────────────────────────────────────┐
    │        Translation / Interpretation       │
    │                                          │
    │  ┌─────────┐ ┌─────────┐ ┌────────────┐ │
    │  │  Music   │ │  TTS    │ │   Effect   │ │
    │  │Retrieval │ │Synthesis│ │  Retrieval │ │
    │  │  Module  │ │ Module  │ │  & Params  │ │
    │  └────┬────┘ └────┬────┘ └─────┬──────┘ │
    └───────┼───────────┼────────────┼────────┘
            │           │            │
            ▼           ▼            ▼
    ┌──────────────────────────────────────────┐
    │           Video Rendering Engine          │
    │  (clips + audio + overlays + effects)    │
    └──────────────────────────────────────────┘
            │
            ▼
      Final Edited Video
```

The pipeline works as follows:

1. **Vidi-Edit** generates the editing plan from raw assets
2. **Music retrieval module** matches music attributes from a curated database
3. **TTS generator** synthesizes voiceover from narration content and speaker attributes
4. **Effect retrieval module** maps visual editing intent to effect primitives
5. **Rendering engine** integrates everything into the final video

### 4.5 Qualitative Examples from the Paper

The paper presents several qualitative examples demonstrating Vidi-Edit's capabilities:

1. **Single-topic (zoo animals)**: Multiple animal clips from a zoo are organized into a coherent narrative with implicit temporal structure — not just visually similar footage concatenated together.

2. **Urban travel**: Heterogeneous city-visit assets (transportation, streets, nightlife, social moments) are unified into a storyline that captures temporal progression and experiential flow.

3. **Sparse inputs (Seattle)**: Each input clip is only a few seconds long, providing minimal temporal context in isolation. The model infers high-level semantic relationships and produces a coherent narrative.

4. **Mixed content**: Semantically diverse assets without obvious temporal or thematic order. Vidi-Edit selectively identifies appropriate clips and constructs a storyline, demonstrating narrative planning beyond low-level visual similarity.

---

## 5. Supporting Capabilities in This Repository

While the Vidi-Edit post-training code is not yet publicly available, this repository contains the foundation models and evaluation tools that underpin the storyline-based video creation system.

### 5.1 Temporal Retrieval

**What it does**: Given a video and a text query, identifies the time ranges that correspond to the query.

**Why it matters for Vidi-Edit**: Temporal retrieval is the core primitive for clip selection and segment extraction. To build a narrative structure, the model must understand *when* specific events, objects, or actions appear in the raw footage.

**Code location**: `Vidi_7B/inference.py`, `Vidi1.5_9B/vidi/eval/inference.py`

**Performance**: On VUE-TR-V2, Vidi2.5 achieves:
- Overall IoU: 49.62% (vs. Gemini 3 Pro at 37.58%, GPT-5 at 17.15%)
- Particularly strong on long (48.54%) and ultra-long (42.22%) videos

### 5.2 Spatio-Temporal Grounding

**What it does**: Given a video and a text query, identifies both the time ranges AND the bounding boxes of target objects within those frames.

**Why it matters for Vidi-Edit**: STG enables fine-grained understanding needed for:
- Automatic view switching (knowing where characters are)
- Composition-aware reframing
- Plot-level understanding of who does what and when

**Performance**: On VUE-STG, Vidi2.5 achieves:
- vIoU: 38.64% (vs. Gemini 3 Pro at 4.61%, GPT-5 at 5.47%)
- This represents a massive lead in spatio-temporal understanding

### 5.3 Plot Understanding and Reasoning

**What it does**: The Vidi2.5-Think model provides:
- Dense speaker localization and speech recognition
- Narrative understanding and character relationship reasoning
- Professional filming and editing technique analysis

**Why it matters for Vidi-Edit**: Plot understanding is essential for:
- Determining which clips are narratively important
- Understanding character arcs and relationships
- Identifying the optimal ordering for storytelling
- Generating appropriate voiceover content

**Performance**: On VUE-PLOT, Vidi2.5-Think achieves:
- Character Track: tIoU 71.63%, sIoU 55.89%, WER 23.20%
- Reasoning Track: Overall accuracy 64.33%

---

## 6. Environment Setup Guide

This repository can be set up on a clean device using Miniconda. A unified setup script is provided.

### Prerequisites

- Linux (Ubuntu 18.04+)
- NVIDIA GPU with CUDA 12.1+ drivers (for model inference)
- Internet connection

### Quick Start

```bash
# Clone the repository
git clone https://github.com/bytedance/vidi.git
cd vidi

# Run the setup script (installs Miniconda if needed)
bash setup_conda.sh all     # Set up all environments
# OR
bash setup_conda.sh vidi7b  # Vidi-7B only
bash setup_conda.sh vidi9b  # Vidi1.5-9B only
bash setup_conda.sh eval    # Evaluation benchmarks only
```

### Manual Conda Setup

```bash
# Install Miniconda (if not already installed)
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
export PATH="$HOME/miniconda3/bin:$PATH"

# For Vidi-7B
conda env create -f Vidi_7B/environment.yml
conda activate vidi7b
pip install "flash-attn==2.6.3" --no-build-isolation

# For Vidi1.5-9B
conda env create -f Vidi1.5_9B/environment.yml
conda activate vidi9b
pip install "flash-attn==2.8.3" --no-build-isolation
```

### Download Model Weights

```bash
# Install huggingface_hub (if not already installed)
pip install huggingface_hub

# Download Vidi-7B
huggingface-cli download bytedance-research/Vidi-7B --local-dir ./models/Vidi-7B

# Download Vidi1.5-9B
huggingface-cli download bytedance-research/Vidi1.5-9B --local-dir ./models/Vidi1.5-9B
```

### Run Inference

```bash
# Vidi-7B (temporal retrieval)
conda activate vidi7b
cd Vidi_7B
python3 -u inference.py \
    --video-path /path/to/video.mp4 \
    --query "a person slicing onions" \
    --model-path /path/to/Vidi-7B

# Vidi1.5-9B (temporal retrieval)
conda activate vidi9b
cd Vidi1.5_9B
python3 -u vidi/eval/inference.py \
    --video-path /path/to/video.mp4 \
    --query "a person slicing onions" \
    --model-path /path/to/Vidi1.5-9B
```

---

## 7. Benchmark and Evaluation

Three evaluation benchmarks are included:

### VUE-TR-V2 (Temporal Retrieval)

```bash
conda activate vue_eval
cd VUE_TR_V2
python3 -u qa_eval.py --pred_path results_Vidi.json
```

- 1,600 queries across 847 videos (310+ hours)
- Covers ultra-short (<1 min) to ultra-long (>60 min) videos
- Metrics: AUC of Precision, Recall, and IoU

### VUE-STG (Spatio-Temporal Grounding)

```bash
cd VUE_STG
python3 evaluate.py
```

- 1,600 queries across 982 videos (204+ hours)
- Evaluates both temporal and spatial localization
- Primary metric: vIoU (spatio-temporal IoU)

### VUE-PLOT (Plot Understanding)

```bash
cd VUE_PLOT
python3 character_eval.py   # Character track
python3 vqa_eval.py          # Reasoning track
```

- Character track: 546 videos, 13,554 speech segments, 33,083 bounding boxes
- Reasoning track: 137 videos, 1,214 QA pairs across 5 reasoning categories

---

## 8. Current Limitations and Availability

### What IS Available in This Repository

| Component | Status | Location |
|-----------|--------|----------|
| Vidi-7B inference | ✅ Available | `Vidi_7B/` |
| Vidi1.5-9B inference | ✅ Available | `Vidi1.5_9B/vidi/eval/` |
| Vidi1.5-9B finetune | ✅ Available | `Vidi1.5_9B/scripts/finetune.sh` |
| Model weights | ✅ On HuggingFace | [Vidi-7B](https://huggingface.co/bytedance-research/Vidi-7B), [Vidi1.5-9B](https://huggingface.co/bytedance-research/Vidi1.5-9B) |
| VUE-TR-V2 benchmark | ✅ Available | `VUE_TR_V2/` |
| VUE-STG benchmark | ✅ Available | `VUE_STG/` |
| VUE-PLOT benchmark | ✅ Available | `VUE_PLOT/` |
| Conda setup | ✅ Available | `setup_conda.sh`, `environment.yml` files |

### What is NOT Yet Available

| Component | Status | Notes |
|-----------|--------|-------|
| Vidi-Edit model weights | ❌ Not released | Post-trained variant for editing plans |
| Vidi-Edit training code | ❌ Not released | Post-training pipeline and data |
| Editing plan execution pipeline | ❌ Not released | Music retrieval, TTS, effect retrieval, rendering |
| Vidi2.5 (12B) weights | ❌ Not released | Full 12B model (only 7B and 9B released) |
| Vidi2.5-Think weights | ❌ Not released | Thinking model for plot reasoning |

### How to Access Vidi-Edit Now

The Vidi-Edit functionality is accessible through the **web demo** at <https://vidi.byteintl.com/>:

1. Navigate to the demo website
2. Select the **"Edit" page**
3. Upload multiple videos
4. Click the generate button
5. The system will automatically output an edited video with storyline, music, effects, etc.

### Potential for Replication

Given the open-sourced components, researchers can partially replicate the Vidi-Edit pipeline:

1. **Use Vidi1.5-9B** for temporal retrieval and video understanding
2. **Fine-tune** using the provided training infrastructure (`scripts/finetune.sh`)
3. **Create editing plan training data** following the specification in Section 4.2
4. **Build the execution pipeline** using open-source tools:
   - TTS: [Coqui TTS](https://github.com/coqui-ai/TTS) or [Bark](https://github.com/suno-ai/bark)
   - Music: [MusicGen](https://github.com/facebookresearch/audiocraft)
   - Video rendering: [MoviePy](https://github.com/Zulko/moviepy) (already a dependency)
   - Visual effects: [FFmpeg](https://ffmpeg.org/) filters

---

## 9. Conclusion

Vidi2.5's Storyline-based Video Creation (Vidi-Edit) represents a significant advance in automated video editing. By separating high-level semantic planning from low-level execution, the system can reason about narrative structure, audio design, and visual effects in a unified framework.

The key innovation is the **editing plan** — a structured representation that captures the full intent of a video edit. This enables:

- **Coherent narratives** from unordered raw footage
- **Multi-clip reasoning** that goes beyond simple visual similarity
- **Coordinated multimodal output** (video, audio, text, effects)
- **Modular execution** where each component can be independently improved

While the Vidi-Edit-specific code and weights are not yet publicly available, the released base models (Vidi-7B, Vidi1.5-9B) demonstrate the strong temporal and spatial understanding that underpins the editing planning capability. The provided conda setup and evaluation benchmarks make it straightforward to experiment with these foundation models on a clean device.

---

## 10. References

1. Vidi2.5 Team. "Vidi2.5: Large Multimodal Models for Video Understanding and Creation." *arXiv preprint arXiv:2511.19529*, 2026.

2. Vidi Team. "Vidi: Large Multimodal Models for Video Understanding and Editing." *arXiv preprint arXiv:2504.15681*, 2025.

3. DeepSeek. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." 2025.

4. OpenAI. "OpenAI o1 System Card." 2024.

5. Google. "Gemma 3 Technical Report." 2025.

6. Alibaba. "Qwen2.5-VL." 2025.

---

*This report was generated based on the arXiv paper (2511.19529) and the open-source code in the Vidi repository. For the latest updates, visit the [project homepage](https://bytedance.github.io/vidi-website/) and the [demo](https://vidi.byteintl.com/).*
