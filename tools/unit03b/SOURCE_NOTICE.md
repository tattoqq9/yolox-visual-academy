# External source and compatibility notice

The Unit 03-B reference implementation is an independent educational rewrite of the components described in:

- Megvii-BaseDetection/YOLOX — https://github.com/Megvii-BaseDetection/YOLOX
- [YOLOX Nano experiment](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/default/yolox_nano.py)
- [Network blocks](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/network_blocks.py)
- [CSPDarknet](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/darknet.py)
- [YOLOPAFPN](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_pafpn.py)
- [YOLOXHead](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py)

The official YOLOX repository is licensed under Apache License 2.0; the license text is available at its upstream LICENSE file. The educational rewrite contains compatible model topology and parameter names to allow strict weights loading, but not the full original training or decoding implementation.

No pretrained weight or checkpoint is included. The bundled trace is generated from randomly initialized parameters and the user's permitted demonstration image; it cannot be used to claim learned head features.

Checkpoint loading is strict but only checks the state-dict schema; it cannot prove that weights are pretrained. The exporter never silently labels an arbitrary user checkpoint as trained.
