# YOLOX Visual Academy v0.6 — Unit 06 research prototype (2026-11-05)

## Added

- **Unit 06**: synthetic 128×128 tiny object playground with stride 4/8/16/32, geometry radius1.5, score thresholds, dynamic K and visually highlighted candidate locations.
- Numerical, client-only demonstration of YOLOX center/exp width decode, Signed LTRB with negative distances and invalid box detection, and an ordered positive-width alternative.
- Research-first reveal gate: student must write a hypothesis before inspecting the signed LTRB comparison. No user hypothesis text is uploaded or stored.
- Japanese step-by-step study notes; 7 intentionally incomplete Python student tasks with tests; YouTube story outline.
- Local-only `tools/unit06/dataset_contract.py` validation-only COCO JSON reader for later formal user-supplied images and labels. PNG/JPEG dimensions, image SHA, label identity/coords/category/ignore/iscrowd, split leakage and safe private manifest storage are checked.
- `tools/unit06/DATASET_CONTRACT_JA.md` explains formal later upgrade of Unit 01–06, rights/privacy/near-duplicate gates, and separate explicit public release.
- GitHub Actions checks Unit 06 Python contract and Node numerical tests before Pages deployment.

## Deliberate limits

- All Unit 06 GT, predicted BBoxes, Objectness and classification logits are **SYNTHETIC illustrative values**, not an evaluated head detection model or benchmark.
- The geometry demo is a **single GT, single stride, one-class** instructional subset. The full official SimOTA combines strides and GTs, resolves conflicts and may differ by implementation revision.
- Signed LTRB is a research hypothesis only. No empirical accuracy advantage is claimed; unconstrained l+r and t+b can yield invalid BBoxes.
- Formal dataset integration is **not done**. This release contains no nonpublic training annotations or model weights. The validator refuses output under the publishable site and marks manifests nonpublishable.
- COCO JSON is the initial accepted label format. Additional label formats are to be determined after the user supplies the formal dataset.
- Browser page tested via inline-asset loading in Chromium because direct localhost/file URLs are prohibited in the validation environment. Public Pages GUI must be validated separately.

## Validation

- Python suite: **90 tests (89 PASS, 1 skipped)**.
- Node numerical tests: **Unit04 11 PASS + Unit06 12 PASS**.
- HTML links: **215 local references in 17 HTML files PASS**.
- Unit06 Chrome desktop/mobile UI: PASS (no JS page errors, no horizontal overflow), using exactly the authored HTML/JS/CSS and embedded identical PNG bytes.
- Public deploy: must be confirmed separately by GitHub Actions after push.

## References

- Megvii-BaseDetection/YOLOX, Apache-2.0, `yolox/models/yolo_head.py` (conceptual topology/formulas; source not copied).
- Prior experimental summaries are not used as numerical evidence of Unit06 performance.
