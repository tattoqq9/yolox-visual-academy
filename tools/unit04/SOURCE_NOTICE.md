# External reference and provenance

- YOLOX (Megvii): https://github.com/Megvii-BaseDetection/YOLOX
- Referenced source: `yolox/models/yolo_head.py` and `yolox/models/losses.py` as viewed 2026-11-04.
- This code is an independently written educational demonstration of formulas and allocation semantics, not copied official code or official checkpoint weights.
- The reference `yolo_head.py` retrieved on 2026-11-04 uses `get_geometry_constraint()` with `center_radius=1.5`, a center-only union candidate filter, pairwise cost = cls_cost + 3*(-log(IoU+1e-8)) + 1e6*invalid, and top-10-sum dynamic K. Older releases may implement an in-box-center combination instead. Compare by a pinned Git SHA when necessary.
- Unit03-C trace source is `site/lab/03-fpn-head/trace.json`; its SHA256 is recorded in Unit04 `trace04.json` metadata. All its weights are seeded random / UNTRAINED.
- No model weights, training examples, private experiment data or ground truth annotations are included in Unit04. Step02–04 use deliberate synthetic examples.
