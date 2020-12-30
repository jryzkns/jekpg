# jekpg

An experimental JPEG inspired image format using JPEG quantization matrices, DCT, and run-length encoding.

For now, only 24bpp bmp file formats are supported. Images need to have dimensions with a multiple of 8.

# Usage

```py
# encoding
py jekpg_enc.py <bmp>


# decoding
py jekpg_dec.py <jekpg>
```
