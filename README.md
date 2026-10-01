# RaFBNet

Official inference package for **RaFBNet: Reliability-Aware Foreground-Background Refinement Network** for salient object detection in optical remote sensing images.

This release contains the inference code and metric summaries. Training code is intentionally not included and will be released separately. Model weights are distributed separately and should be placed under `weights/`.

## Contents

```text
RaFBNet/
  data.py
  models/
    rafbnet.py
    rafbnet_blocks.py
    smt_tiny/
      smt.py
      loader.py
  tools/
    infer.py
  weights/
    README.md
  metrics/
    results.json
```

## Checkpoints and results

RaFBNet uses an **SMT-Tiny** hierarchical encoder together with frozen RADIO v2.5-B semantic priors, reliability gating, and foreground-background refinement. The current paper reports eight evaluation metrics on ORSSD, EORSSD, and ORSI-4199. Use the dataset-specific checkpoints to reproduce the corresponding results:

| Dataset | Checkpoint | $S_\alpha$ | $F_\beta^{max}$ | $F_\beta^{mean}$ | $F_\beta^{adp}$ | $E_\xi^{max}$ | $E_\xi^{mean}$ | $E_\xi^{adp}$ | MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ORSSD | `weights/RaFBNet_ORSSD.pth` | 0.9535 | 0.9290 | 0.9163 | 0.9082 | 0.9869 | 0.9831 | 0.9824 | 0.0059 |
| EORSSD | `weights/RaFBNet_EORSSD.pth` | 0.9453 | 0.8990 | 0.8875 | 0.8717 | 0.9856 | 0.9805 | 0.9797 | 0.0046 |
| ORSI-4199 | `weights/RaFBNet_ORSI4199.pth` | 0.8904 | 0.8914 | 0.8864 | 0.8846 | 0.9597 | 0.9529 | 0.9564 | 0.0243 |

Download the checkpoint bundle from Baidu Netdisk:

```text
Link: https://pan.baidu.com/s/1LwK35HxuSiPGFJvbzUTcVw
Extraction code: dfcc
Archive: RaFB.zip
SHA-256: 57140e8faa27b2feec50d32e1a4f29779e1f4471ef16d451f0cbcc9fe9ee74e0
```

The bundle contains:

```text
RaFBNet_ORSSD.pth
RaFBNet_EORSSD.pth
RaFBNet_ORSI4199.pth
```

Place them under `weights/` before running inference.

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
  --image_root ./dataset/ors-4199/testset/images \
  --gt_root ./dataset/ors-4199/testset/gt \
  --checkpoint ./weights/RaFBNet_ORSI4199.pth \
  --save_root ./outputs/ORSI-4199 \
  --radio_repo /path/to/RADIO \
  --radio_checkpoint /path/to/radio-v2.5-b_half.pth.tar
```

Predicted saliency maps are written as normalized grayscale PNG files, one per test image. The metric summaries in `metrics/results.json` are produced by evaluating these maps with the `py_sod_metrics` toolkit on the standard test splits.

## Notes

- This package is inference-only.
- Dataset files are not included.
- Put the downloaded checkpoint files under `weights/` before running inference.
- The GitHub repository itself does not contain model checkpoints; checkpoint hosting is handled separately because each RaFBNet checkpoint is approximately 443 MB.
- The code is named consistently as RaFBNet to avoid coupling the public release to earlier internal experiment names.
