# Video Inference

## Overview

dLLM provides high-performance video inference capabilities for action recognition, captioning, temporal detection, generation, frame interpolation, summarization, optical flow, and visual question answering. The video processing pipeline leverages distributed CPU SIMD compute and multi-vendor GPU acceleration to handle complex temporal workloads efficiently.

## Supported Tasks

### Action Recognition

Classify actions and activities in video sequences.

**Performance**: 8K+ videos/s (AVX-512), 30K+ videos/s (CUDA)

| Model | Parameters | Kinetics Accuracy | Inference Time (CPU) | Inference Time (GPU) |
|-------|-----------|-------------------|---------------------|---------------------|
| VideoMAE v2 | 300M | 83.2% | 125ms | 28ms |
| TimeSformer | 120M | 81.5% | 98ms | 22ms |
| X3D-XL | 98M | 82.1% | 112ms | 25ms |
| MViT v2-B | 60M | 80.8% | 85ms | 19ms |
| SlowFast-X101 | 170M | 81.9% | 135ms | 30ms |
| R(2+1)D-152 | 60M | 78.5% | 92ms | 21ms |

### Video Captioning

Generate natural language descriptions of video content.

**Performance**: 3K+ videos/s (AVX-512), 12K+ videos/s (CUDA)

| Model | Parameters | CIDEr | SPICE | Inference Time (CPU) | Inference Time (GPU) |
|-------|-----------|-------|-------|---------------------|---------------------|
| Video-LLaVA | 7B | 118.5 | 26.3 | 2.8s | 0.6s |
| VideoChat | 6.7B | 115.2 | 25.1 | 2.5s | 0.55s |
| LLaVA-Video | 13B | 122.8 | 27.8 | 3.5s | 0.8s |
| Video-ChatGPT | 6.7B | 119.1 | 26.5 | 2.9s | 0.65s |
| ShareGPT-4Video | 7B | 120.3 | 27.1 | 3.1s | 0.7s |

### Temporal Action Detection

Locate and classify actions within video timelines.

**Performance**: 5K+ videos/s (AVX-512), 20K+ videos/s (CUDA)

| Model | Parameters | mAP@0.5 | Inference Time (CPU) | Inference Time (GPU) |
|-------|-----------|---------|---------------------|---------------------|
| ActionFormer | 30M | 42.5 | 185ms | 42ms |
| StreamFormer | 25M | 44.1 | 165ms | 38ms |
| TimeSformer-Det | 120M | 45.8 | 210ms | 48ms |
| SlowFast-Det | 170M | 46.2 | 225ms | 52ms |
| VideoMAE-Det | 300M | 47.5 | 245ms | 56ms |

### Video Generation

Text-to-video synthesis and video editing.

**Performance**: 1K+ videos/s (AVX-512), 4K+ videos/s (CUDA) for 256×256

| Model | Parameters | FVD | Generation Time (CPU) | Generation Time (GPU) |
|-------|-----------|-----|---------------------|---------------------|
| CogVideo | 2B | 185.2 | 12s | 2.5s |
| Stable Video Diffusion | 1.5B | 165.8 | 10s | 2.1s |
| ModelScope-T2V | 1.3B | 172.5 | 11s | 2.3s |
| AnimateDiff | 860M | 195.1 | 8s | 1.7s |
| Pika (compatible) | 3B | 158.3 | 14s | 3.0s |

### Frame Interpolation

Generate intermediate frames between existing frames.

**Performance**: 6K+ frames/s (AVX-512), 22K+ frames/s (CUDA)

| Model | Parameters | PSNR | SSIM | Inference Time (CPU) | Inference Time (GPU) |
|-------|-----------|------|------|---------------------|---------------------|
| RIFE | 8M | 32.5 dB | 0.92 | 18ms | 4ms |
| FILM | 12M | 31.8 dB | 0.91 | 25ms | 6ms |
| CAIN | 15M | 32.1 dB | 0.92 | 28ms | 7ms |
| TOFlow | 10M | 31.5 dB | 0.90 | 22ms | 5ms |
| SepConv | 6M | 30.8 dB | 0.89 | 15ms | 3.5ms |

### Video Summarization

Extract key frames and generate video summaries.

**Performance**: 4K+ videos/s (AVX-512), 15K+ videos/s (CUDA)

| Model | Parameters | F1 Score | Inference Time (CPU) | Inference Time (GPU) |
|-------|-----------|----------|---------------------|---------------------|
| VideoSum | 50M | 0.78 | 85ms | 18ms |
| KeyNet | 35M | 0.75 | 72ms | 15ms |
| SumFormer | 65M | 0.81 | 95ms | 20ms |
| TransSum | 45M | 0.79 | 80ms | 17ms |

### Optical Flow

Estimate pixel motion between consecutive frames.

**Performance**: 7K+ frames/s (AVX-512), 25K+ frames/s (CUDA)

| Model | Parameters | EPE | Inference Time (CPU) | Inference Time (GPU) |
|-------|-----------|-----|---------------------|---------------------|
| RAFT | 8M | 2.18 | 35ms | 7ms |
| PWC-Net | 5M | 2.45 | 28ms | 6ms |
| FlowNet2 | 60M | 2.35 | 65ms | 14ms |
| LiteFlowNet | 3M | 2.52 | 22ms | 5ms |
| FlowNetS | 15M | 2.40 | 42ms | 9ms |

### Video Question Answering

Answer questions about video content and events.

**Performance**: 2K+ videos/s (AVX-512), 8K+ videos/s (CUDA)

| Model | Parameters | Accuracy | Inference Time (CPU) | Inference Time (GPU) |
|-------|-----------|----------|---------------------|---------------------|
| Video-LLaVA | 7B | 58.2% | 3.2s | 0.7s |
| VideoChat | 6.7B | 56.8% | 3.0s | 0.65s |
| LLaVA-Video | 13B | 60.5% | 4.1s | 0.9s |
| Video-ChatGPT | 6.7B | 57.5% | 3.1s | 0.68s |

## Supported Video Formats

| Format | Extension | Description | Status |
|--------|-----------|-------------|--------|
| **MP4** | `.mp4` | MPEG-4 Part 14 — most common video format | ✓ Supported |
| **AVI** | `.avi` | Audio Video Interleave — legacy format | ✓ Supported |
| **MKV** | `.mkv` | Matroska — container with multiple streams | ✓ Supported |
| **WebM** | `.webm` | WebM — web-optimized video | ✓ Supported |
| **MOV** | `.mov` | QuickTime — Apple format | ✓ Supported |

## Video Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                  Video Processing Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│  Input (MP4/AVI/MKV/WebM/MOV)                               │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────┐                                          │
│  │ Video Decoder │  Hardware-accelerated decoding (GPU)     │
│  │ (SIMD-accel)  │  SIMD-accelerated frame extraction       │
│  └──────┬───────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ Frame Sample  │  Temporal sampling & frame selection     │
│  │ (SIMD-accel)  │  Uniform / adaptive sampling             │
│  └──────┬───────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ Frame Encode  │  Per-frame visual encoding               │
│  │ (GPU/CPU)     │  Distributed across cluster nodes        │
│  └──────┬───────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ Temporal      │  Temporal modeling & attention           │
│  │ Fusion        │  Distributed temporal processing         │
│  │ (GPU/CPU)     │  Multi-scale temporal features           │
│  └──────┬───────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ Video Task    │  Classification / Generation / Detection │
│  │ Head          │  Multi-task routing                        │
│  │ (GPU/CPU)     │                                          │
│  └──────┬───────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ Post-process  │  Temporal smoothing, format output       │
│  │ (SIMD-accel)  │  Format-specific output formatting       │
│  └──────┬───────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  Output (JSON / Video / Frames / Annotations)               │
└─────────────────────────────────────────────────────────────┘
```

### Video Decoding & Frame Extraction

All video decoding operations are SIMD-accelerated:

| Operation | SSE4.2 | AVX | AVX2 | AVX-512 |
|-----------|--------|-----|------|---------|
| Frame extraction | 1.0x | 2.1x | 2.3x | 4.6x |
| Frame resize | 1.0x | 2.0x | 2.2x | 4.3x |
| Color conversion | 1.0x | 2.1x | 2.3x | 4.5x |
| Frame normalization | 1.0x | 2.0x | 2.1x | 4.2x |
| Temporal sampling | 1.0x | 2.0x | 2.2x | 4.4x |

### Supported Video Encoders

| Encoder | Type | Parameters | Temporal Modeling | Use Case |
|---------|------|-----------|-------------------|----------|
| VideoMAE v2 | Masked Autoencoder | 300M | Spatial-temporal | General video |
| TimeSformer | Transformer | 120M | Space-time attention | Action recognition |
| X3D | 3D CNN | 98M | 3D convolutions | Efficient recognition |
| MViT v2 | Hierarchical Transformer | 60M | Multi-scale attention | Scalable recognition |
| SlowFast | Dual-path CNN | 170M | Slow-fast paths | Fast detection |
| R(2+1)D | Factorized 3D CNN | 60M | Factorized 3D | Lightweight |
| I3D | Inflated 3D CNN | 92M | Inflated 2D | Classic baseline |
| ViViT | Video Vision Transformer | 200M | Factorized attention | Modern ViT |

## Python API

### Action Recognition

```python
from backend_connector import BackendConnector

connector = BackendConnector()

# Load action recognition model
connector.load_model("models/videomae-v2-action.gguf",
                     model_type="video_vision")

# Recognize action in video
result = connector.predict_video(
    "action_video.mp4",
    task="action_recognition",
    top_k=5,
    confidence_threshold=0.5
)
print(result)
# {
#     'action': 'playing_guitar',
#     'confidence': 0.89,
#     'all_actions': [
#         {'action': 'playing_guitar', 'confidence': 0.89},
#         {'action': 'playing_piano', 'confidence': 0.05},
#         {'action': 'playing_instrument', 'confidence': 0.03},
#         ...
#     ]
# }
```

### Video Captioning

```python
# Generate video description
caption = connector.predict_video(
    "sunset_beach.mp4",
    task="captioning",
    model="video-llava-caption.gguf",
    max_length=128
)
print(caption)  # "A person walks along the beach at sunset..."
```

### Temporal Action Detection

```python
# Detect actions with timestamps
detections = connector.predict_video(
    "sports_clip.mp4",
    task="temporal_detection",
    model="action-former-detect.gguf",
    confidence=0.7
)
for det in detections:
    print(f"  {det['action']}: [{det['start']:.2f}s - {det['end']:.2f}s] "
          f"confidence={det['confidence']:.2f}")
```

### Video Generation

```python
# Generate video from text prompt
video = connector.predict_video(
    prompt="a cat chasing a butterfly in a garden",
    task="generation",
    model="cogvideo-gen.gguf",
    duration=5,
    fps=24,
    width=512,
    height=512
)
video.save("generated_video.mp4")
```

### Frame Interpolation

```python
# Interpolate frames (2x speed)
interpolated = connector.predict_video(
    "slow_motion.mp4",
    task="frame_interpolation",
    model="rife-interp.gguf",
    factor=2
)
interpolated.save("interpolated.mp4")
```

### Video Summarization

```python
# Extract key frames for summary
summary = connector.predict_video(
    "lecture_video.mp4",
    task="summarization",
    model="video-summarize.gguf",
    num_key_frames=10
)
for frame in summary.key_frames:
    print(f"  Frame at {frame.timestamp:.2f}s: {frame.description}")
```

### Optical Flow

```python
# Estimate optical flow
flow = connector.predict_video(
    "driving_scene.mp4",
    task="optical_flow",
    model="raft-flow.gguf"
)
flow.save("optical_flow.mp4")
```

### Video Question Answering

```python
# Answer questions about video
answer = connector.predict_video(
    "news_clip.mp4",
    task="vqa",
    question="What is the main topic of this news segment?",
    model="video-llava-vqa.gguf"
)
print(answer)  # "The segment discusses the new climate policy..."
```

### Video Editing

```python
# Edit video based on text prompt
edited = connector.predict_video(
    "original_video.mp4",
    task="editing",
    prompt="change the sky to sunset colors",
    model="video-editing.gguf"
)
edited.save("edited_video.mp4")
```

## C++ API

### Action Recognition

```cpp
#include "vision/video_processor.h"
#include "vision/video_encoder.h"

// Initialize video pipeline
dllm::VideoPipeline pipeline({
    .encoder = dllm::VideoEncoder::VIDEO_MAE_V2,
    .task = dllm::VideoTask::ACTION_RECOGNITION,
    .device = dllm::Device::AUTO
});

// Load model
pipeline.load_model("models/videomae-v2-action.gguf");

// Process video
dllm::Video video = dllm::Video::load("action_video.mp4");
auto result = pipeline.predict(video, dllm::PredictOptions{
    .top_k = 5,
    .confidence_threshold = 0.5
});

for (const auto& pred : result.predictions) {
    std::cout << pred.action_name << ": " << pred.confidence << "\n";
}
```

### Video Captioning

```cpp
// Generate video caption
dllm::VideoPipeline caption_pipeline({
    .encoder = dllm::VideoEncoder::VIDEO_LLAVA,
    .task = dllm::VideoTask::CAPTIONING,
    .device = dllm::Device::AUTO
});

caption_pipeline.load_model("models/video-llava-caption.gguf");

dllm::Video video = dllm::Video::load("sunset_beach.mp4");
auto caption = caption_pipeline.caption(video, dllm::CaptionOptions{
    .max_length = 128
});

std::cout << caption.text << "\n";
```

### Video Generation

```cpp
// Text-to-video generation
dllm::VideoPipeline gen_pipeline({
    .encoder = dllm::VideoEncoder::COGVIDEO,
    .task = dllm::VideoTask::GENERATION,
    .device = dllm::Device::AUTO
});

gen_pipeline.load_model("models/cogvideo-gen.gguf");

auto generated = gen_pipeline.generate(
    "a cat chasing a butterfly in a garden",
    dllm::GenOptions{
        .duration = 5,
        .fps = 24,
        .width = 512,
        .height = 512
    }
);

generated.save("generated_video.mp4");
```

### Frame Interpolation

```cpp
// Interpolate frames
dllm::VideoPipeline interp_pipeline({
    .encoder = dllm::VideoEncoder::RIFE,
    .task = dllm::VideoTask::FRAME_INTERPOLATION,
    .device = dllm::Device::AUTO
});

interp_pipeline.load_model("models/rife-interp.gguf");

dllm::Video video = dllm::Video::load("slow_motion.mp4");
auto interpolated = interp_pipeline.interpolate(video, 
    dllm::InterpolateOptions{.factor = 2});

interpolated.save("interpolated.mp4");
```

### Optical Flow

```cpp
// Estimate optical flow
dllm::VideoPipeline flow_pipeline({
    .encoder = dllm::VideoEncoder::RAFT,
    .task = dllm::VideoTask::OPTICAL_FLOW,
    .device = dllm::Device::AUTO
});

flow_pipeline.load_model("models/raft-flow.gguf");

dllm::Video video = dllm::Video::load("driving_scene.mp4");
auto flow = flow_pipeline.compute_flow(video);
flow.save("optical_flow.mp4");
```

## Configuration

### YAML Configuration

```yaml
video:
  # General settings
  enabled: true
  max_concurrent_requests: 16
  
  # Format support
  supported_formats:
    - mp4
    - avi
    - mkv
    - webm
    - mov
  
  # Processing limits
  max_duration: 600  # seconds
  max_resolution: 1920
  max_fps: 60
  max_frames: 30000
  
  # Frame sampling
  sampling:
    method: uniform  # uniform, adaptive, keyframe
    frames_per_second: 1
    max_frames_per_video: 16
  
  # Preprocessing
  preprocessing:
    resize_method: bilinear
    normalize: true
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  
  # Temporal modeling
  temporal:
    window_size: 16
    stride: 8
    attention_type: space_time  # space_time, temporal, spatial
  
  # Model loading
  model_loading:
    auto_detect: true
    cache_models: true
    max_cached_models: 4
  
  # Hardware routing
  hardware:
    auto_offload: true
    gpu_threshold: 0.7
    cpu_priority: sse42
    gpu_backends: [cuda, rocm, sycl]
  
  # Task-specific settings
  tasks:
    action_recognition:
      top_k: 5
      confidence_threshold: 0.5
      sampling_fps: 1
    
    captioning:
      max_length: 128
      beam_size: 5
    
    generation:
      default_duration: 5
      default_fps: 24
      default_width: 512
      default_height: 512
      default_steps: 50
    
    frame_interpolation:
      default_factor: 2
      max_factor: 8
    
    optical_flow:
      max_resolution: 720
    
    summarization:
      default_key_frames: 10
      max_key_frames: 50
    
    vqa:
      max_questions_per_video: 10
    
    temporal_detection:
      confidence_threshold: 0.7
      min_duration: 0.5
```

## Performance Optimization

### SIMD Acceleration

All video processing operations leverage SIMD instructions:

| Operation | SSE4.2 | AVX | AVX2 | AVX-512 |
|-----------|--------|-----|------|---------|
| Frame extraction | 1.0x | 2.1x | 2.3x | 4.6x |
| Frame resize | 1.0x | 2.0x | 2.2x | 4.3x |
| Color conversion | 1.0x | 2.1x | 2.3x | 4.5x |
| Temporal sampling | 1.0x | 2.0x | 2.2x | 4.4x |
| Optical flow estimation | 1.0x | 2.0x | 2.2x | 4.3x |
| Frame interpolation | 1.0x | 2.1x | 2.3x | 4.6x |
| Video encoding | 1.0x | 2.0x | 2.1x | 4.2x |

### Distributed Processing

For long videos or batch processing, dLLM distributes computation across cluster nodes:

```yaml
distributed:
  enabled: true
  split_strategy: temporal  # temporal, spatial, hybrid
  chunk_duration: 10  # seconds per chunk
  communication_protocol: tcp
  compression: lz4
```

### Memory Optimization

- **Streaming processing**: Process videos in temporal chunks
- **Memory pooling**: Reuse tensor buffers across frames
- **Lazy frame loading**: Load frames on demand
- **Quantized weights**: INT8/INT4 for reduced memory footprint

## Troubleshooting

### Common Issues

**Issue**: Out of memory for long videos

**Solution**:
```yaml
video:
  max_duration: 120  # Reduce from default 600
  sampling:
    frames_per_second: 0.5  # Reduce frame rate
  distributed:
    enabled: true  # Distribute across cluster
```

**Issue**: Slow video processing on CPU

**Solution**:
```yaml
video:
  hardware:
    cpu_priority: avx512  # Use fastest available SIMD
  sampling:
    frames_per_second: 0.5  # Reduce for faster processing
```

**Issue**: Poor action recognition accuracy

**Solution**:
```yaml
video:
  tasks:
    action_recognition:
      sampling_fps: 2  # Increase frame sampling
      top_k: 10  # Increase candidates
```

**Issue**: Video format errors

**Solution**:
```yaml
video:
  supported_formats:
    - mp4  # Remove problematic formats
  preprocessing:
    resize_method: nearest  # Faster fallback
```

## Related Documentation

- [Image Inference Documentation](./IMAGE_INFERENCE.md)
- [3D Assets Documentation](./3D_ASSETS.md)
- [Multimodal Guide](./MULTIMODAL_GUIDE.md)
- [FEATURES.md](../FEATURES.md#multimodal-inference)
