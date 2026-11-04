# YOLOX Visual Academy v0.3-B — Release notes (2026-11-04)

## Added
- Unit 03-B interactive, rotating 3D multi-channel feature map display for the **real torch forward pass** of a 416×416 **YOLOX-Nano-compatible topology**.
- Focus → dark2 → dark3 → dark4 → dark5: output tensor shapes, strides, stage-level parameter counts, selectable channels and pixel values.
- Standard Conv vs depthwise+pointwise parameter comparison with controllable C_in/C_out/K.
- Actual CSP branch outputs (dark3.1 conv1, bottleneck path, conv2, concatenation, conv3), not the simplified concept from Unit03-A.
- Independent PyTorch model reference with 80-class parameter count **912,159**, optional official-checkpoint loading, deterministic trace exporter. No checkpoint or pretrained weight included.
- Unit 03-B textbook, PyTorch student starter, numerical unit tests, and video storyboard.
- Static JSON trace schema verification in GitHub Actions; original Units 01–03-A preserved.

## Crucial provenance warning
- **The bundled 1.95 MB trace is an actual torch forward pass with seeded random-initialized weights**, not a trained feature extractor. It does not demonstrate learned head detectors or semantic saliency.
- Public trace contains only sampled channels. Other channels exist in the actual model but are not serialized.
- Preprocess: RGB selfie → 416×416 upper-left letterbox (RGB=114 padding), PIL bilinear resize, BGR float32 values 0..255, not divided by 255. OpenCV reproduction can differ slightly.
- The reference `YOLOX` class matches published component shapes/parameterization but is an educational rewrite. The official training pipeline, SimOTA loss, NMS, and postprocessing are **not** implemented here.
- Student exercises depend on a separate local PyTorch installation; no PyTorch required to use public website.

## Next: Unit 03-C
Full PAFPN and Decoupled Head dataflow with official trained features only once checkpoint license/provenance and local checks pass; no false feature semantics.
