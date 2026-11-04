# YOLOX Visual Academy v0.3-C — 2026-11-04

## Added
- Unit03-C: PAFPN Top-down, Bottom-up, Decoupled Head interactive tracer with 38 real computed PyTorch nodes, selectable channels and grid coordinates.
- Tensor display with rotatable 3D channel stack; exact Upsample and Concat provenance comparisons; head raw 4+1+80 outputs, sigmoid explanation and 1-class hypothetical output shape.
- New textbook, PyTorch incomplete student starter and automated tests, video storyboard.
- `tools/unit03c/export_neck_head.py` deterministic trace generation using the same Unit03-B reference model and preprocessing.
- Python standard-library trace integrity checks in CI; existing Units 01–03-B retained.
- Extended public HTTP verification script to cover 28 endpoints including Unit03-C assets.

## Model and dataset provenance
- Model corresponds to official YOLOX-Nano topology (depth=0.33, width=0.25, depthwise=True), 80 classes, input 416×416.
- **Weights: UNTRAINED fixed-seed random initialization**, not pretrained. Trace is a genuine PyTorch forward with random weights and must not be described as learned object/head features.
- Input: authorized site sample `003Z-selfie.png`, preprocessed with Pillow letterbox to BGR 0..255 float32. Unit03-B preview is reused.
- Only selected channels are serialized in the 1.8MB `trace.json`; others exist inside PyTorch but are not web-provided.
- This is an independently written educational reference; official YOLOX code is not bundled.

## Verified metrics
- YOLOX-Nano (80 classes): 912,159 parameters, 459,968 backbone + 347,424 neck + 104,767 head.
- P3 64×52×52, P4 128×26×26, P5 256×13×13; total 3,549 positions and 301,665 raw scalars.
- Exact Upsample value replication and Concat channel boundary checks. Head raw `[reg4,obj1,cls80]` order verified.

## Remaining gates
- Browser smoke in local test environment; final live Pages GUI click-through by user on Windows/WSL is separate.
- Training, NMS, SimOTA and loss, official checkpoint licensing, beginner review, and narrated YouTube video remain future work.
