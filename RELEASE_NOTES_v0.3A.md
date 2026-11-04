# v0.3-A release notes

- Adds experimental Unit 03-A (Conv, per-image spatial-statistic BN, SiLU, simplified CSP).
- Uses 003Z-selfie.png, resampled to 16×12 grayscale entirely within browser.
- Adds student code starter ZIP, teaching document and numeric reference tests.
- CI now runs new Python unit tests and syntax check.
- Keeps Unit01/02 paths and content preserved in this packaged release.

Known limits: not the actual YOLOX-Nano structure; BN uses one image rather than batch or running stats, CSP demonstrates branch/transform/concat only; full browser GUI automation not yet passed in this execution environment.
