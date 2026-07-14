# DotMetrics TEM Detector Weights

This repository releases the trained YOLOv8n detector used by DotMetrics for
nanoparticle localization in transmission electron microscopy (TEM) images.

> The detector returns particle bounding boxes. DotMetrics performs its
> particle-size measurement through a separate contour-based segmentation and
> scale-calibration workflow.

## Contents

| File | Description |
| --- | --- |
| `yolov8n-DotMetrics.pt` | Released YOLOv8n detector weights. |
| `predict.py` | Command-line inference script. |
| `train.py` | Training-script template using the recorded run settings. |
| `training/train8-args.yaml` | Settings recorded for the run that produced the released weights. |
| `data-template.yaml` | Dataset configuration template. |

## Quick start

Create a Python environment, install the dependency, and run inference on a
TEM image (for example, a `.tif`, `.png`, or `.jpg` file):

```bash
pip install -r requirements.txt
python predict.py --source path/to/tem_image.tif
```

Predictions are written to `runs/detect/predict/` by default. The script uses
an image size of 1280 pixels, confidence threshold of 0.15, and IoU threshold
of 0.30 by default. These values can be changed from the command line:

```bash
python predict.py --source path/to/tem_image.tif --conf 0.15 --iou 0.30 --save-txt
```

## Model information

| Item | Value |
| --- | --- |
| Architecture | Ultralytics YOLOv8n |
| Task | Object detection |
| Detection classes | 1 (`cds`) |
| Base checkpoint | `yolov8n.pt` |
| Training image size | 1280 |
| Training schedule | 500 epochs, batch size 12, AdamW optimizer |
| Random seed | 0 |
| Weight file SHA-256 | `DB994F722D05FBA38498894F30F3103F5E15A99684EA1549EF699D1DC23CE9C1` |

The run settings above were recovered from the original `train8` training
record that produced `yolov8n-DotMetrics.pt`.

## Re-training

Raw TEM images and annotations are not distributed in this repository. Create
a local dataset YAML from `data-template.yaml`, then run:

```bash
python train.py --data path/to/data.yaml
```

The script reads `training/train8-args.yaml` and uses its recorded settings.
You are responsible for ensuring that any training images and annotations you
use may be redistributed and used for machine-learning research.

## Limitations

This weight file was developed for the TEM nanoparticle-imaging setting
described in the associated DotMetrics work. Performance may change with
particle material, imaging conditions, magnification, contrast, or image
preprocessing. Validate the model on representative images before using its
outputs for scientific measurements.

PyTorch `.pt` files use Python serialization. Load model weights only from
trusted sources.

## License

The code and released modified model weights are provided under the
[GNU Affero General Public License v3.0](LICENSE). This repository uses
Ultralytics YOLO; users must also comply with the applicable Ultralytics
license terms.

## Citation

Please cite the associated DotMetrics manuscript when using these weights in
research. Full bibliographic information will be added after publication.
