# MIGC (Multi-Instance Generation Controller for Text-to-Image Synthesis)

A tool for controlled image generation with multiple elements and spatial positioning.

## Required Files
- MIGC_SD14.ckpt
- cetusMix_Whalefall2.safetensors
- pytorch_model.bin (CLIP text encoder)

## Quick Start
```python
prompt_final = [['masterpiece, best quality, blue colored sofa', 'big muscle black man']]
bboxes = [[[0.3, 0.2, 0.8, 0.7]]]

image = pipe(
    prompt_final,
    bboxes,
    num_inference_steps=20,
    guidance_scale=7.5,
    MIGCsteps=15,
    NaiveFuserSteps=30
).images[0]
```

## Key Parameters
- `num_inference_steps`: Denoising steps
- `guidance_scale`: Prompt adherence strength
- `MIGCsteps`: Generation control steps
- `NaiveFuserSteps`: Fusion process steps

## Note
- Coordinates use 0-1 range
- First prompt describes overall scene
- Additional prompts match with bounding boxes