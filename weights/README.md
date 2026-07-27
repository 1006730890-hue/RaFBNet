# Weights

This no-weight GitHub package does not include checkpoint files.

The current paper-aligned checkpoint bundle should contain the following files. Download them separately, then place them here:

- `RaFBNet_ORSSD.pth`
- `RaFBNet_EORSSD.pth`
- `RaFBNet_ORSI4199_epoch46.pth`
- `pvt_v2_b2.pth`

Download the bundle from Baidu Netdisk:

```text
Link: https://pan.baidu.com/s/1urZk3k0EWJPAvwL28YnMvg
Extraction code: erh5
Archive: RaFBNet_paper_aligned_weights_windows_20260727.zip
SHA-256: cf2a3a4ff3946212b8099fe911936e5a97d5a67f179bc76b1cdedc60305ef6ad
```

RADIO v2.5-B is an external dependency. Download or place the RADIO checkpoint separately and pass its path via `--radio_checkpoint`.
