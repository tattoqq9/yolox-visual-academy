# YOLOX Visual Academy v0.4 — Unit 04 prototype (2026-11-04)

## Added

- 4-stage browser lab: actual **UNTRAINED** P3/P4/P5 raw → YOLOX grid/stride BBox Decode, synthetic IoU+NMS, synthetic 2-class SimOTA, synthetic mini-batch Loss and gradients.
- A compact extracted trace (`site/lab/04-detection/trace04.json`) containing the true recorded tx/ty/tw/th/obj/class0 channels from previous Unit03-C PyTorch forward. SHA256 of source trace recorded in provenance.
- Controllable grid coordinates and hypothetical BBox raw edits, with visible 'hypothetical' label after change.
- IoU comparison of 4/16/80px boxes, class-aware NMS selection, SimOTA radius1.5 current `get_geometry_constraint` variant, top10 dynamic K, candidate/GT conflict resolution.
- BCEWithLogits, 1−IoU², quality target and analytic logit gradient vs finite-difference BBox gradient. Neither NN backprop nor actual training is claimed.
- Detailed Japanese textbook, Python stdlib TODO/auto-tests, math.js with Node regression, standalone Python reference and data integrity tests, YouTube storyboard.

## Scientific and rights boundary

- Step01 only: **fixed-seed UNTRAINED model head raw values**; no trained head detector inference.
- Step02–04: explicitly **SYNTHETIC** boxes, GT labels and candidates designed to expose the mathematics. Not annotations to selfie and not published experimental results.
- Official YOLOX `yolo_head.py` is a moving `main` branch. On 2026-11-04 the viewed reference uses `get_geometry_constraint` radius1.5 for SimOTA. Older versions may differ; pin a commit for future scientific replicability.
- Current unit implements a numerically faithful instructional subset, not a full trainer. Score calibration, learned feature saliency, dataset metrics, model weights and formal beginner studies are out of scope.

## Next

- Public site after-action GUI verification and beginner-review feedback.
- Unit 05: scratch build + train + infer + score, with model/weights and rights kept separate.
