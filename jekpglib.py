import numpy as np
from collections import namedtuple

c = namedtuple('Config',
    [
        'Q_CONST',
        'uv_subsample',
        'dct',
        'quant',
        'rle'
    ])

config = c(
    Q_CONST      = 10,
    uv_subsample = False,
    quant        = True,
    dct          = True,
    rle          = True
)

def rpad_len(w, bpp=24): return (4-((bpp>>3)*w)%4)%4

def f_fetch_int(file_handle, n, sign = False):
    return int.from_bytes(
        file_handle.read(n),
        byteorder="little",
        signed=sign)

def readbmp(in_path):
    with open(in_path, "rb") as fp:

        signature       = fp.read(2)
        file_size       = f_fetch_int(fp,4)
        reserved1       = f_fetch_int(fp,2)
        reserved2       = f_fetch_int(fp,2)
        data_offset     = f_fetch_int(fp,4)
        info_hdr_size   = f_fetch_int(fp,4)
        bmp_w           = f_fetch_int(fp,4, sign=True)
        bmp_h           = f_fetch_int(fp,4, sign=True)
        planes          = f_fetch_int(fp,2)
        bpp             = f_fetch_int(fp,2)
        compression     = f_fetch_int(fp,4)
        image_size      = f_fetch_int(fp,4)
        x_pix_per_m     = f_fetch_int(fp,4, sign=True)
        y_pix_per_m     = f_fetch_int(fp,4, sign=True)
        colors_used     = f_fetch_int(fp,4)
        importantcolors = f_fetch_int(fp,4)

        bmp_dat, row_pad_len = [], rpad_len(bmp_w)
        for _ in range(bmp_h):
            for _ in range(bmp_w):
                bmp_dat.append(tuple(fp.read(bpp >> 3))[::-1])
            fp.read(row_pad_len)

        fp.seek(0); header = fp.read(54)
        return bmp_dat, bmp_w, bmp_h, header

def asbytes(v, n, byo = 'little'):
    return v.to_bytes(n, byteorder = byo)

def writebmp(fn, data, w, h, header):
    i, row_pad_len = 0, rpad_len(w)
    with open(fn, "wb") as out_f:
        out_f.write(header)
        for _ in range(h):
            for _ in range(w):
                for val in data[i]:
                    out_f.write(asbytes(int(val), 1))
                i += 1
            out_f.write(row_pad_len* b'\0')

m_r2y = (   ( 1/4, 1/2,  1/4),
            ( 1/2,   0, -1/2),
            (-1/4, 1/2, -1/4))

m_y2r = (   ( 1,  1, -1),
            ( 1,  0,  1),
            ( 1, -1, -1))

def dot(u,v): return sum(map(lambda x : x[0]*x[1], zip(u,v)))
def rgb2ycocg(rgb): return np.array([dot(rgb,r) for r in m_r2y])
def ycocg2rgb(yuv): return np.array([dot(yuv,r) for r in m_y2r])

def subsample2x(m): return m[::2, ::2]
def  upsample2x(m): return np.kron(m, np.ones((2, 2)))

def gen_dct_mat(N):
    return np.array([
        [np.sqrt(2/N) * np.cos((2*j + 1) * i * np.pi/(2*N))
            for j in range(N)]
                if i != 0 else [ 1 / np.sqrt(N) ] * N
                    for i in range(N)])

m_dcts = {  4 : gen_dct_mat(4),
            8 : gen_dct_mat(8)}

Q50 = np.array(( ( 16,  11,  10,  16,  24,  40,  51,  61),
                 ( 12,  12,  14,  19,  26,  58,  60,  55),
                 ( 14,  13,  16,  24,  40,  57,  69,  56),
                 ( 14,  17,  22,  29,  51,  87,  80,  62),
                 ( 18,  22,  37,  56,  68, 109, 103,  77),
                 ( 24,  35,  55,  64,  81, 104, 113,  92),
                 ( 49,  64,  78,  87, 103, 121, 120, 101),
                 ( 72,  92,  95,  98, 112, 100, 103,  99)))

Q50_4 = np.array((  (16, 11, 10, 16),
                    (12, 12, 14, 19),
                    (14, 13, 16, 24),
                    (14, 17, 22, 29)))

m_quants = {    4 : np.multiply( Q50_4, config.Q_CONST),
                8 : np.multiply(   Q50, config.Q_CONST)}

def zz_scan(N): # This assumes a NxN grid

    seq, cx, cy = [(0, 0), (1, 0)], 1, 0
    while len(seq) < N**2:

        # go southwest until it can't
        while (cx > 0 and cy + 1 < N):
            cx, cy = cx - 1, cy + 1
            seq.append((cx, cy))

        # go down or right
        if (cy + 1 < N): cy += 1
        else:            cx += 1
        seq.append((cx, cy))

        if len(seq) == N**2: break

        # go northeast until it can't
        while(cy > 0 and cx  + 1 < N):
            cx, cy = cx + 1, cy - 1
            seq.append((cx, cy))

        # go right or down
        if (cx + 1 < N): cx += 1
        else:            cy += 1
        seq.append((cx, cy))

    return seq

zz_scans = {    4 : zz_scan(4),
                8 : zz_scan(8)}

def RLE_enc(seq):
    out = []
    dur, curr = 1, seq[0]
    for sym in seq[1:]:
        if sym == curr: dur += 1
        else:
            out.append((dur, curr))
            dur, curr = 1, sym
    if curr != 0: out.append((dur, curr))
    return tuple(out)

def RLE_dec(seq):
    return tuple([ val
        for (freq, val) in seq
        for _ in range(freq)])
