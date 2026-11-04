# YOLOX Visual Academy — roadmap

## Status (2026-11-04)

- **Unit 01**: RGB → 3×3 Conv → feature maps — published prototype
- **Unit 02**: stride → receptive field → feature map grid → geometric assignment candidates — published prototype
- **Unit 03-A**: Conv → BatchNorm → SiLU → simplified CSP — published prototype, learner review pending
- **Unit 03-B**: official YOLOX-Nano inference topology, Focus/DWConv/CSPLayer, parameter counts and **real PyTorch tensor data from seeded random weights** — published prototype, learner review pending
- **Unit 03-C (next)**: YOLOPAFPN and Decoupled Head; full multiscale tensor flow, output labels
- Unit 04: BBox decode / SimOTA / loss and gradients
- Unit 05: YOLOX-Nano scratch implementation and evaluation
- Unit 06: tiny heads and signed LTRB design research
- Unit 1.0: beginner QA, narrative teaching videos, model/asset rights clearances, polished documentation

## Release gates

- Local relative links pass `scripts/check_site.py`
- CI and optional CPU torch numerical checks pass
- JS syntax and GUI interaction tests pass
- Geometry and tensor values are numerically grounded; trained vs untrained status is explicit
- Input image and third-party model/weights provenance reviewed
- All existing Units remain unchanged and working
- HTTP 200 and actual in-browser interaction confirmed after GitHub Pages deploy

## Environment boundaries

This repository is the sole target. Do not edit, stop, build, or reconfigure the user's head detector R&D and Tatto Engine repos. Public site is static and works without a backend or external image transmission.
