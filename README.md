# RaFBNet

Official inference package for **RaFBNet: Reliability-aware Foreground-Background Network** for salient object detection in optical remote sensing images.

This GitHub-ready release contains inference code, reported saliency maps, and metric summaries. Training code is intentionally not included and will be released separately. Model weights are not included in this no-weight package; download them separately from Baidu Netdisk and place them under `weights/`.

## Contents

```text
RaFBNet_release_no_weights/
  data.py
  models/
    rafbnet.py
    rafbnet_blocks.py
    pvtv2.py
  tools/
    infer.py
  weights/
    README.md
  predictions/
    ORSSD/
    EORSSD/
  metrics/
    metrics_ORSSD.json
    metrics_EORSSD.json
```

## Checkpoints

Download the following files separately from Baidu Netdisk, then place them in `weights/`:

Baidu Netdisk:

```text
Link: https://pan.baidu.com/s/1iafDNRJd3HnfgP38IXutLw
Extraction code: iv42
Archive: weights.zip
```

| Dataset | Checkpoint | Epoch | S-measure | wF-measure | MAE |
|---|---:|---:|---:|---:|---:|
| ORSSD | `weights/RaFBNet_ORSSD_epoch47.pth` | 47 | 0.948672 | 0.912539 | 0.006864 |
| EORSSD | `weights/RaFBNet_EORSSD_epoch51.pth` | 51 | 0.943145 | 0.889039 | 0.004627 |

The PVTv2-B2 backbone checkpoint should also be placed at:

```text
weights/pvt_v2_b2.pth
```

Recommended Baidu Netdisk bundle:

```text
RaFBNet_ORSSD_epoch47.pth
RaFBNet_EORSSD_epoch51.pth
pvt_v2_b2.pth
```

## External RADIO dependency

RaFBNet uses frozen RADIO v2.5-B dense features as semantic reliability priors. To run inference, clone or install the RADIO repository separately and provide:

- `--radio_repo`: path to the RADIO repository containing `hubconf.py`
- `--radio_checkpoint`: path to the RADIO v2.5-B checkpoint, e.g. `radio-v2.5-b_half.pth.tar`

The RADIO source and checkpoint are third-party dependencies and are not redistributed in this package.

## Installation

```bash
conda create -n rafbnet python=3.10
conda activate rafbnet
pip install -r requirements.txt
```

Install PyTorch matching your CUDA version from the official PyTorch instructions if the default requirement does not match your environment.

## Inference

Example on ORSSD:

```bash
python tools/infer.py \
  --image_root ./dataset/test_dataset/ORSSD/image \
  --gt_root ./dataset/test_dataset/ORSSD/GT \
  --checkpoint ./weights/RaFBNet_ORSSD_epoch47.pth \
  --save_root ./outputs/ORSSD \
  --radio_repo /path/to/RADIO \
  --radio_checkpoint /path/to/radio-v2.5-b_half.pth.tar
```

Example on EORSSD:

```bash
python tools/infer.py \
  --image_root ./dataset/test_dataset/EORSSD/image \
  --gt_root ./dataset/test_dataset/EORSSD/GT \
  --checkpoint ./weights/RaFBNet_EORSSD_epoch51.pth \
  --save_root ./outputs/EORSSD \
  --radio_repo /path/to/RADIO \
  --radio_checkpoint /path/to/radio-v2.5-b_half.pth.tar
```

Predicted saliency maps are written as normalized grayscale PNG files.

## Notes

- This package is inference-only.
- Dataset files are not included.
- This no-weight package is suitable for normal GitHub upload without Git LFS.
- Put downloaded checkpoint files under `weights/` before running inference.
- The code is named consistently as RaFBNet to avoid coupling the public release to earlier internal experiment names.
