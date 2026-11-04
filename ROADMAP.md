# YOLOX Visual Academy — Roadmap

## Status: 2026-11-04

- **Unit 01**: RGB → Conv → feature maps — published prototype
- **Unit 02**: stride → receptive field → grids — published prototype
- **Unit 03-A**: Conv → BatchNorm → SiLU → introductory CSP — published prototype
- **Unit 03-B**: Nano Backbone, DWConv, real untrained PyTorch feature trace — published prototype
- **Unit 03-C**: YOLOPAFPN, Head branches, 38 nodes of untrained trace — published prototype
- **Unit 04**: BBox Decode, IoU, NMS, SimOTA, Loss and gradients — educational prototype; browser and beginner review gate pending
- **Unit 05 next**: YOLOX-Nano scratch model and training/evaluation
- **Unit 06 planned**: tiny objects, diagnose positive assignment and signed LTRB ideation
- **v1.0 planned**: beginner QA, YouTube narration, licensing and accessibility

## Release policy

- All Units remain playable; current site and 28+ public HTTP endpoints must pass
- `python3 scripts/check_site.py site` and `python3 -m unittest discover -s tests -v` must PASS
- `node --test tests/unit04_math.test.cjs` must PASS
- Local browser interaction tests on PC and phone must PASS; publicly served GUI check separate
- Numerical formulas are grounded in official YOLOX `yolo_head.py`/`losses.py`, with variant note
- Actual untrained forward trace and synthetic GT examples labeled accurately
- `main` updated only after local PASS and production deploy validated by GitHub Actions
- Copyright, image source, dataset/weight license, and source attribution audited

## Protected boundaries

This repository is the only target. Do not edit, stop, build, commit or reconfigure the user's head detection R&D or Tatto Engine repositories. Educational site stays static, no API, uploads or arbitrary external user-image transmission.
