# RaFBNet

Official inference package for **RaFBNet: Reliability-Aware Foreground-Background Refinement Network** for salient object detection in optical remote sensing images.

This GitHub-ready release contains inference code, saliency maps, and metric summaries. Training code is intentionally not included and will be released separately. Model weights are distributed separately and should be placed under `weights/`.

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
    results.json
```

## Paper-aligned checkpoints and results

The current paper reports eight evaluation metrics on ORSSD, EORSSD, and ORSI-4199. Use the following dataset-specific checkpoints to reproduce the corresponding results:

| Dataset | Checkpoint | $S_\alpha$ | $F_\beta^{max}$ | $F_\beta^{mean}$ | $F_\beta^{adp}$ | $E_\xi^{max}$ | $E_\xi^{mean}$ | $E_\xi^{adp}$ | MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ORSSD | `weights/RaFBNet_ORSSD.pth` | 0.9496 | 0.9262 | 0.9140 | 0.9056 | 0.9860 | 0.9816 | 0.9827 | 0.0063 |
| EORSSD | `weights/RaFBNet_EORSSD.pth` | 0.9439 | 0.8997 | 0.8857 | 0.8701 | 0.9850 | 0.9801 | 0.9788 | 0.0045 |
| ORSI-4199 | `weights/RaFBNet_ORSI4199_epoch46.pth` | 0.8888 | 0.8904 | 0.8860 | 0.8855 | 0.9567 | 0.9499 | 0.9534 | 0.0263 |

Download the paper-aligned checkpoint bundle from Baidu Netdisk:

```text
Link: https://pan.baidu.com/s/1urZk3k0EWJPAvwL28YnMvg
Extraction code: erh5
Archive: RaFBNet_paper_aligned_weights_windows_20260727.zip
SHA-256: cf2a3a4ff3946212b8099fe911936e5a97d5a67f179bc76b1cdedc60305ef6ad
```

The PVTv2-B2 backbone checkpoint should also be placed at:

```text
weights/pvt_v2_b2.pth
```

The paper-aligned checkpoint bundle should contain:

```text
RaFBNet_ORSSD.pth
RaFBNet_EORSSD.pth
RaFBNet_ORSI4199_epoch46.pth
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
  --checkpoint ./weights/RaFBNet_ORSSD.pth \
  --save_root ./outputs/ORSSD \
  --radio_repo /path/to/RADIO \
  --radio_checkpoint /path/to/radio-v2.5-b_half.pth.tar
```

Example on EORSSD:

```bash
python tools/infer.py \
  --image_root ./dataset/test_dataset/EORSSD/image \
  --gt_root ./dataset/test_dataset/EORSSD/GT \
  --checkpoint ./weights/RaFBNet_EORSSD.pth \
  --save_root ./outputs/EORSSD \
  --radio_repo /path/to/RADIO \
  --radio_checkpoint /path/to/radio-v2.5-b_half.pth.tar
```

Example on ORSI-4199:

```bash
python tools/infer.py \
  --image_root ./dataset/test_dataset/ORSI-4199/image \
  --gt_root ./dataset/test_dataset/ORSI-4199/GT \
  --checkpoint ./weights/RaFBNet_ORSI4199_epoch46.pth \
  --save_root ./outputs/ORSI-4199 \
  --radio_repo /path/to/RADIO \
  --radio_checkpoint /path/to/radio-v2.5-b_half.pth.tar
```

Predicted saliency maps are written as normalized grayscale PNG files.

## Notes

- This package is inference-only.
- Dataset files are not included.
- This no-weight package is suitable for normal GitHub upload without Git LFS.
- Put downloaded checkpoint files under `weights/` before running inference.
- The GitHub repository itself does not contain model checkpoints; checkpoint hosting is handled separately because each RaFBNet checkpoint is approximately 497 MB.
- The code is named consistently as RaFBNet to avoid coupling the public release to earlier internal experiment names.
