# YOLOX Visual Academy — roadmap

## Status (2026-11-04)

- Unit 01: RGB → 3×3 Conv → feature maps: published
- Unit 02: stride → receptive field → feature map grid → geometric assignment candidate: published
- Unit 03-A: Conv → BatchNorm → SiLU → simplified CSP: prototype for learner review (not official YOLOX intermediate tensors)
- Unit 03-B: YOLOX-Nano actual layers / depthwise and standard convolution / parameter counts / actual intermediate tensors: next
- Unit 03-C: PAFPN and decoupled head: planned
- Unit 04: Box decode / SimOTA / loss and gradients: planned
- Unit 05: YOLOX-Nano full scratch implementation and evaluation: planned
- Unit 06: tiny heads and signed LTRB design challenges: planned
- Unit 1.0: beginner user review, teaching videos, licensing, published documentation: planned

## Release gates

All site-relative URLs pass check_site.py, numerical tests pass, JavaScript syntax passes, all assets are rights-cleared and complete, desktop/mobile interactions manually checked, and deployment HTTP + browser interactions pass.

## Scope boundary

The site is independent from tatto-engine and head detector R&D repos. No modifications to those repos. The tutorial computation is deliberately simplified and is not the pretrained YOLOX model.
