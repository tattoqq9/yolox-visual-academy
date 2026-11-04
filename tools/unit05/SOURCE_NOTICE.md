# Unit 05 code and data provenance

The `core.py`, `train.py`, `evaluate.py`, and `dataset.py` implementations are independently written for educational use, using a YOLOX-Nano-compatible topology from `tools/unit03b/nano_reference.py`. The architecture is based on [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0).

- Dataset: locally generated artificial yellow rectangles on 128×128 images.
- Training: PyTorch autograd + optimizer, single GT/image, simplified YOLOX geometry matching and loss; **not the official YOLOX training recipe**.
- Evaluation: one-class AP50 on six synthetic examples, with single-class NMS. This is **not COCO mAP or human-head detection performance**.
- No third-party weights, private dataset, or real-person photos are used in training.
- Checkpoints must stay in `out/` and must not be included in public site files.
