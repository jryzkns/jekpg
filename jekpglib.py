import numpy as np

def f_fetch_int(file_handle, n, sign=False):
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
        
        bitmap_data, row_pad_len = [], (4 - ((bpp >> 3) * bmp_w) % 4) % 4
        for _ in range(bmp_h):
            for _ in range(bmp_w):
                bitmap_data.append(tuple(fp.read(bpp >> 3))[::-1])
            fp.read(row_pad_len)

        fp.seek(0); header = fp.read(54)
        return bitmap_data, bmp_w, bmp_h, header

def dot(u,v):
    return sum(map(lambda x : x[0] * x[1], zip(u, v)))

m_rgb2ycocg = (   
    (1/4,   1/2,    1/4),
    (1/2,   0,      -1/2),
    (-1/4,  1/2,    -1/4))
                  
def rgb2ycocg(rgb):
    return np.array([dot(rgb, row) for row in m_rgb2ycocg])

m_ycocg2rgb = (   
    (1,     1,     -1),
    (1,     0,     1),
    (1,     -1,    -1))

def ycocg2rgb(ycocg):
    return np.array([dot(ycocg, row) for row in m_ycocg2rgb])

def subsample2x(m):
    return m.reshape(
        (m.shape[0] >> 1, 2, m.shape[1] >> 1, 2)
            ).mean(-1).mean(1)

def upsample2x(m):
    return np.kron(m, np.ones((2, 2)))

def gen_dct_mat(N):
    return np.array([
        [np.sqrt(2/N) * np.cos((2*j + 1) * i * np.pi/(2*N)) 
            for j in range(N)] 
                if i != 0 else [ 1 / np.sqrt(N) ] * N 
                    for i in range(N)])

m_dct_8x8 = gen_dct_mat(8)
m_dct_4x4 = gen_dct_mat(4)

Q50 = np.array(( 
    ( 16,  11,  10,  16,  24,  40,  51,  61),
    ( 12,  12,  14,  19,  26,  58,  60,  55),
    ( 14,  13,  16,  24,  40,  57,  69,  56),
    ( 14,  17,  22,  29,  51,  87,  80,  62),
    ( 18,  22,  37,  56,  68, 109, 103,  77),
    ( 24,  35,  55,  64,  81, 104, 113,  92),
    ( 49,  64,  78,  87, 103, 121, 120, 101),
    ( 72,  92,  95,  98, 112, 100, 103,  99)))

Q50_4 = np.array((  
    (16, 11, 10, 16),
    (12, 12, 14, 19),
    (14, 13, 16, 24),
    (14, 17, 22, 29)))

def zz_scan(N): # This assumes a NxN grid
    
    sequence = [(0,0), (1,0)]
    curr_x, curr_y = 1, 0
    while len(sequence) < N**2:

        # go southwest until it can't
        while (curr_x > 0 and curr_y + 1 < N):
            curr_x, curr_y = curr_x - 1, curr_y + 1
            sequence.append((curr_x, curr_y))

        # go down or right
        if (curr_y + 1 < N): curr_y += 1
        else:                curr_x += 1
        sequence.append((curr_x, curr_y))

        if len(sequence) == N**2: break

        # go northeast until it can't
        while(curr_y > 0 and curr_x  + 1 < N):
            curr_x, curr_y = curr_x + 1, curr_y - 1
            sequence.append((curr_x, curr_y))
        
        # go right or down
        if (curr_x + 1 < N): curr_x += 1
        else:                curr_y += 1
        sequence.append((curr_x, curr_y))

    return sequence

# TODO: Why are we casting to int here?
def RLE_enc(seq):
    out = []
    curr, duration = int(seq[0]), 1
    duration, curr = 1, int(seq[0])
    for sym in seq[1:]:
        if sym == curr:
            duration += 1
        else:
            out.append((duration, curr))
            duration, curr = 1, int(sym)
    if curr != 0:
        out.append((duration, curr))
    return tuple(out)

def RLE_dec(seq):
    return tuple(
        [val for (freq, val) in seq 
                for _ in range(freq)])
