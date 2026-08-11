# EmotionCLIP-FT Track-B mapping

## Identity and source

- Paper/release: EmotionCLIP, CVPR 2023.
- Official repository: `https://github.com/Xeaver/EmotionCLIP`.
- Fixed commit: `ca142dd4c9664acb2b59c8b14ef3169049f1180b`.
- Fixed source snapshot tree SHA-256:
  `c313af40432b83d4ef41700fb216de2ddecc9ecb882390d939e0b0c9690f79a7`.
- License: MIT.
- Official pretrained checkpoint: the Google Drive asset linked on line 39 of
  the fixed upstream README (`file id 1-p2i5peK2zgf3grK-aJUdfEwUdW9-53Z`).
- External read-only source directory: `baseline_sources/emotionclip_release_ca142dd`.
  The upstream repository is not copied into this benchmark.

The upstream EMOTIC number, `32.91` mAP, is a static frozen-feature result. It
uses L2-normalized EmotionCLIP features and one-vs-rest logistic regression
with `C=2.5`. It is contextual evidence, not an incremental benchmark number.

## Method-FT conversion

EmotionCLIP-FT is a Track-B static-to-incremental control. It retains the
native subject-aware ViT-B/32 visual encoder and official EmotionCLIP
initialization. The EMOTIC image is resized by its short side to 224 with
bicubic interpolation, center-cropped, ImageNet-normalized, and paired with the
center-cropped binary target-person bbox mask. The mask creates the additional
subject token exactly as in the source visual operator.

At each B5-C3 task the method appends a linear block for the new classes and
fine-tunes the complete visual encoder and the newly appended current-task
head; prior head blocks receive no gradient because old labels are hidden.
Training uses sigmoid BCE only on current-class labels. Validation selects the earliest
epoch with maximum current-class mAP. Inference concatenates task heads in
protocol order.

This conversion intentionally has no anti-forgetting mechanism:

- no old-class or future-class ground truth;
- no distillation, replay, EWC or Adapter;
- no CLIP text encoder or text features;
- no test-set threshold or checkpoint selection.

The fixed first validation configuration is 25 maximum epochs, patience 5,
AdamW with source betas `(0.98, 0.9)` and epsilon `1e-6`, visual LR `1e-5`,
head LR `1e-4`, weight decay `1e-4`, gradient clip 1.0, AMP and TF32 enabled,
train batch 64, evaluation batch 128, and global reporting threshold 0.5. The
learning rates and current-label BCE follow the benchmark's registered full-FT
control because the upstream EMOTIC protocol trains only a static sklearn
classifier and therefore supplies no EMOTIC neural fine-tuning optimizer.

## Reproducibility gates

The official full checkpoint is mandatory. Generic OpenAI/OpenCLIP weights and
silent scratch fallback are rejected. Each run records the checkpoint SHA-256.
The preflight verifies fixed upstream file hashes and compares a small source
VisualTransformer to the independent port with maximum absolute error at most
`1e-7`. A CUDA smoke measures the actual batch-64 optimizer-step memory before
validation begins.

The seed-0 validation launcher creates the universal download package at:

`$RUN_ROOT/download_packages/$RUN_ID.tar.gz`

with an adjacent `.sha256`. The package includes canonical scores, metrics,
manifests, HTML and logs, and explicitly excludes every `.pth` checkpoint.
