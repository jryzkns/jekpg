# jekpg

An experimental JPEG inspired image format.

For now, only 24bpp bitmap file formats are supported for input, and the output format is in 24bpp bitmap as well.

input needs to have dimensions with a multiple of 8.

# Features

- Block-based parallelism using multiprocessing in Encoder
- 420 chroma channel subsampling (feature is under maintenance)
- JPEG quantization
- DCT

# Usage

```py
# encoding
py jekpg_enc.py <bmp>

# decoding
py jekpg_dec.py <jekpg>
```
