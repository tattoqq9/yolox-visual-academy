# YOLOX Visual Academy — Roadmap

## Status: 2026-11-05

- **Unit 01** RGB → Convolution → feature maps — published prototype.
- **Unit 02** Stride → receptive field → grids — published prototype.
- **Unit 03-A** Conv → BatchNorm → SiLU → introductory CSP — published prototype.
- **Unit 03-B** Nano Backbone, Depthwise, actual untrained PyTorch trace — published prototype.
- **Unit 03-C** YOLOPAFPN, decoupled Head, 38-node untrained trace — published prototype.
- **Unit 04** BBox Decode, IoU, NMS, SimOTA, Loss & gradients — published educational prototype.
- **Unit 05** One-class YOLOX-Nano, actual CPU training on 1/8 procedurally generated rectangle images, separate validation images, AP50, checkpoint restoration, interactive browser trace — reconstructed v0.5 prototype; public post-deploy GUI review pending.
- **Unit 06** Synthetic tiny-BBox geometry, score/assignment diagnostics, signed LTRB ideation, validation-only COCO importer — prototype implemented; post-deploy GUI review pending.
- **v1.0 planned** Beginner QA, narrated YouTube series, license provenance and accessibility audit. **After course completion:** user-provided formal images and labels reviewed, converted and integrated only with explicit rights approval.

## Scientific gates

- Unit05 actual experimental results from reconstruction differ from preserved prior-session notes. Do not claim recovery of original training checkpoints or numerical replication.
- Single-class synthetic rectangle AP50 is not COCO mAP or real head-detection performance.
- Track train-vs-eval discrepancies, including BatchNorm statistics and low score, before scientific conclusions.
- Students must distinguish genuine PyTorch weight updates from browser playback of the resulting trace.
- New `main` commits only after local regression PASS, static asset checks and browser smoke. GitHub Actions deploy and public GUI check are separate gates.

## Protected boundaries

Only `tattoqq9/yolox-visual-academy` is in scope; do not write to head-detection R&D, Voxel Engine, or their shared settings/processes.

## Future formal data upgrade gate

- Existing synthetic lesson stays usable, and formal data will be a separate opt-in track after Units 01–06 are reviewed.
- Local-only COCO validation produces unpublishable manifests (SHA-256, bbox/category/ignore audit, split leakage checks). No automatic uploads or dataset promotion.
- Public examples require explicit image/label/model license and privacy review, plus new reproduced weights/trace/metrics and full UI regression.
