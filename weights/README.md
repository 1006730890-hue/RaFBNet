# Weights

This GitHub package does not include checkpoint files (each RaFBNet checkpoint is about 443 MB).

Download the checkpoint bundle separately and place the three files here:

- `RaFBNet_ORSSD.pth`     -> ORSSD
- `RaFBNet_EORSSD.pth`    -> EORSSD
- `RaFBNet_ORSI4199.pth`  -> ORSI-4199

Download the bundle from Baidu Netdisk:

```text
Link: https://pan.baidu.com/s/1LwK35HxuSiPGFJvbzUTcVw
Extraction code: dfcc
Archive: RaFB.zip
SHA-256: 57140e8faa27b2feec50d32e1a4f29779e1f4471ef16d451f0cbcc9fe9ee74e0
```

RADIO v2.5-B is an external dependency. Place the RADIO checkpoint separately and pass its path via
`--radio_checkpoint`; pass the RADIO source directory via `--radio_repo`.
