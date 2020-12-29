import jekpglib

import threading
from multiprocessing import get_context
import multiprocessing
import pickle
import numpy as np

BSIZE = 8
N_WORKERS = 10
QUALITY_CONST = 5
zz_scans = {
    4 : jekpglib.zz_scan(4), 
    8 : jekpglib.zz_scan(8)}

class jekEncoder:

    def __init__(self, fn):

        self.fn = fn
        self.img, self.w, self.h, self.header = jekpglib.readbmp(fn)
        self.cid = {"y" : 0, "co" : 1, "cg" : 2}

    def encode(self):

        # covert 1D RGB values into separate YCoCg channels and unflatten to 2D
        self.y, self.co, self.cg = map( 
            lambda x : np.array(x).reshape((self.w, self.h)),
                zip(*[jekpglib.rgb2ycocg(pix) for pix in self.img]))

        # apply subsampling on chroma
        self.co, self.cg = map(jekpglib.subsample2x, (self.co, self.cg))

        self.channels = {"y" : self.y, "co" : self.co, "cg" : self.cg}
        all_blks =    self.gen_blk_info(*self.y.shape,  "y")  \
                    + self.gen_blk_info(*self.co.shape, "co") \
                    + self.gen_blk_info(*self.cg.shape, "cg")

        with open(self.fn[:-4] + ".jekpg", "wb") as f__:
            f__.write(self.header)
            with get_context("spawn").Pool(N_WORKERS) as worker:
                for (ch, b, x1, y1), data in worker.map(self.blk_job, all_blks):
                    f__.write(ch.to_bytes(1, byteorder='little'))
                    f__.write( b.to_bytes(1, byteorder='little'))
                    f__.write(x1.to_bytes(2, byteorder='little'))
                    f__.write(y1.to_bytes(2, byteorder='little'))
                    datalen = len(data).to_bytes(3, byteorder='little')
                    f__.write(datalen)
                    f__.write(data)

    def blk_job(self, meta):
        
        ch, x1, y1, x2, y2 = meta

        d , N   = self.channels[ch][ x1 : x2, y1 : y2 ].copy(), y2 - y1
        m_dct   = (jekpglib.m_dct_8x8 if N == 8 else jekpglib.m_dct_4x4).copy()
        m_quant = np.multiply((jekpglib.Q50 if N == 8 else jekpglib.Q50_4).copy(), QUALITY_CONST)

        d = m_dct.dot(d).dot(m_dct.T)
        d = np.round_(np.divide(d, m_quant))
        d = [ d[zigzag_step] for zigzag_step in zz_scans[N] ]
        d = jekpglib.RLE_enc(d)

        return (self.cid[ch], int(N == 8), x1, y1), pickle.dumps(d)

    def gen_blk_info(self, w, h, ch):

        naive_info = [( j, i, j + BSIZE if (j + BSIZE) < w else w,
                              i + BSIZE if (i + BSIZE) < h else h)
            for i in range(0, h, BSIZE) for j in range(0, w, BSIZE)]
        
        out = [] # split rectangular blocks into square blocks
        for x1, y1, x2, y2 in naive_info:
            bw, bh = (x2 - x1), (y2 - y1)
            if (bw == bh):  # square block
                out.append(    (ch, x1,   y1,   x2,   y2))
            else:
                if bw < bh: #4x8 case
                    out.append((ch, x1,   y1,   x2,   y1+4))
                    out.append((ch, x1,   y1+4, x2,   y2))
                else:       #8x4 case
                    out.append((ch, x1,   y1,   x1+4, y2))
                    out.append((ch, x1+4, y1,   x2,   y2))

        return out

class jekDecoder:

    def __init__(self, fn):

        self.fn = fn
        with open(self.fn, "rb") as f__:
            
            self.outheader = f__.read(54)
            self.bitstream = f__.read()

            f__.seek(18)
            self.w = int.from_bytes(f__.read(4), byteorder="little", signed=True)
            self.h = int.from_bytes(f__.read(4), byteorder="little", signed=True)

        self.y  = np.zeros(shape=(self.w,      self.h))
        self.co = np.zeros(shape=(self.w >> 1, self.h >> 1))
        self.cg = np.zeros(shape=(self.w >> 1, self.h >> 1))
        self.channels = {0 : self.y, 1 : self.co, 2 : self.cg}

    def read_stream(self, n):
        out = self.bitstream[:n]
        self.bitstream = self.bitstream[n:]
        return out

    def read_blocks(self):

        self.blocks = []
        while self.bitstream != b'':
            ch      = int.from_bytes(self.read_stream(1), byteorder="little")
            btype   = int.from_bytes(self.read_stream(1), byteorder="little")
            x1      = int.from_bytes(self.read_stream(2), byteorder="little")
            y1      = int.from_bytes(self.read_stream(2), byteorder="little")
            datalen = int.from_bytes(self.read_stream(3), byteorder="little")
            data    = pickle.loads(self.read_stream(datalen))
            self.blocks.append(((ch, btype, x1, y1), data))

    def decode(self):

        self.read_blocks()
        for i, ((ch, btype, x1, y1), data) in enumerate(self.blocks):

            N       = (8 if btype == 1 else 4)
            block   = np.zeros((N, N))
            x2, y2  = x1 + N, y1 + N
            m_quant = (jekpglib.Q50 if N == 8 else jekpglib.Q50_4).copy()
            m_dct   = (jekpglib.m_dct_8x8 if N == 8 else jekpglib.m_dct_4x4).copy()

            # unflatten coeffs and undo quantization
            flat_coeffs = jekpglib.RLE_dec(data)
            for coeff, (i,j) in zip(flat_coeffs, zz_scans[N]):
                block[i,j] = coeff * (m_quant[i][j] * QUALITY_CONST)

            # invert DCT and write value into channel buffer
            block = np.round_((m_dct.T).dot(block).dot(m_dct))
            self.channels[ch][x1:x2, y1:y2] = block.copy()

        # upsample chroma channels to match luminance
        self.co, self.cg = map(jekpglib.upsample2x, (self.co, self.cg))
        
        # finally generate bgr values
        self.decoded_bgr = [(jekpglib.ycocg2rgb(ycocg).clip(0,255)[::-1])
                for ycocg in np.dstack((self.y,self.co,self.cg)).reshape((self.w * self.h, 3))]
        self.write_bmp()

    def write_bmp(self):
        i, row_pad_len = 0, (4 - ((self.w * 3) % 4)) % 4
        with open(self.fn[:-4] + "_DEC.bmp", "wb") as out_f:
            out_f.write(self.outheader)
            for _ in range(self.h):
                for _ in range(self.w):
                    for val in self.decoded_bgr[i]:
                        out_f.write(int(val).to_bytes(1, byteorder="little"))
                    i += 1
                out_f.write(row_pad_len* b'\0')