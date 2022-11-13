import jekpglib

import threading
from multiprocessing import get_context
import multiprocessing
import pickle
import numpy as np

import sys

BSIZE = 8
N_WORKERS = 20
Q_CONST = 5

Y, Co, Cg = 0, 1, 2

class jekEncoder:

    def __init__(self, fn):

        self.fn = fn
        self.img, self.w, self.h, self.header = jekpglib.readbmp(fn)

    def encode(self):

        self.y, self.co, self.cg = map(
            lambda x : np.array(x).reshape((self.w, self.h)),
                zip(*[jekpglib.rgb2ycocg(pix) for pix in self.img]))

        if jekpglib.config.uv_subsample:
            self.co, self.cg = map(
                jekpglib.subsample2x, (self.co, self.cg))

        self.channels = {Y : self.y, Co : self.co, Cg : self.cg}
        all_blks = [ 
                        *self.gen_blk_info( *self.y.shape,  Y),
                        *self.gen_blk_info(*self.co.shape, Co),
                        *self.gen_blk_info(*self.cg.shape, Cg),
        ]

        out_fn = self.fn[:-4] + ".jekpg"
        with open(out_fn, "wb") as f__:
            f__.write(self.header)
            with get_context("spawn").Pool(N_WORKERS) as worker:
                worker.daemon = True
                for (ch, b, x1, y1), data in worker.map(self.enc_blk_job, all_blks):
                    f__.write(jekpglib.asbytes(       ch, 1))
                    f__.write(jekpglib.asbytes(        b, 1))
                    f__.write(jekpglib.asbytes(       x1, 2))
                    f__.write(jekpglib.asbytes(       y1, 2))
                    f__.write(jekpglib.asbytes(len(data), 3))
                    f__.write(data)

        return out_fn

    def gen_blk_info(self, w, h, ch):

        naive_info = [( 
            j, 
            i,
            (j + BSIZE) if (j + BSIZE) < w else w,
            (i + BSIZE) if (i + BSIZE) < h else h)
                for i in range(0, h, BSIZE)
                for j in range(0, w, BSIZE)]

        out = [] # split rectangular blocks into square blocks
        for x1, y1, x2, y2 in naive_info:
            bw, bh = (x2 - x1), (y2 - y1)
            if (bw == bh):  # square block
                out.append(    (ch,   x1,   y1,   x2,   y2))
            else:
                if bw < bh: # 4x8 case
                    out.append((ch,   x1,   y1,   x2, y1+4))
                    out.append((ch,   x1, y1+4,   x2,   y2))
                else:       # 8x4 case
                    out.append((ch,   x1,   y1, x1+4,   y2))
                    out.append((ch, x1+4,   y1,   x2,   y2))

        return out

    def enc_blk_job(self, meta):

        ch, x1, y1, x2, y2 = meta

        d, N = self.channels[ch][ x1 : x2, y1 : y2 ], y2 - y1

        if jekpglib.config.dct:
            m_dct = jekpglib.m_dcts[N]
            d = m_dct.dot(d).dot(m_dct.T)

        if jekpglib.config.quant:
            m_quant = jekpglib.m_quants[N]
            d = np.divide(d, m_quant)

        d = [ d[zigzag_step] for zigzag_step in jekpglib.zz_scans[N] ]

        if jekpglib.config.rle:
            d = jekpglib.RLE_enc(d)

        return (ch, int(N == 8), x1, y1), pickle.dumps(d)

class jekDecoder:

    def __init__(self, fn):

        self.fn = fn
        with open(self.fn, "rb") as f__:

            self.outheader = f__.read(54)
            self.bitstream = f__.read()

            f__.seek(18)
            self.w = int.from_bytes(f__.read(4), byteorder="little", signed=True)
            self.h = int.from_bytes(f__.read(4), byteorder="little", signed=True)

        y_dim = (self.w, self.h)
        if jekpglib.config.uv_subsample:
            co_dim = (self.w >> 1, self.h >> 1)
            cg_dim = (self.w >> 1, self.h >> 1)
        else:
            co_dim, cg_dim = y_dim, y_dim

        self.y  = np.zeros(shape =  y_dim)
        self.co = np.zeros(shape = co_dim)
        self.cg = np.zeros(shape = cg_dim)

        self.channels = {Y : self.y, Co : self.co, Cg : self.cg}

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
            data    =   pickle.loads(self.read_stream(datalen))
            self.blocks += [((ch, btype, x1, y1), data)]

    def decode(self):

        self.read_blocks()

        for (ch, btype, x1, y1), data in self.blocks:

            N      = (8 if btype == 1 else 4)
            block  = np.zeros((N, N))
            x2, y2 = x1 + N, y1 + N

            if jekpglib.config.rle:
                data = jekpglib.RLE_dec(data)

            if jekpglib.config.quant:
                m_quant = jekpglib.m_quants[N]

            for coeff, (i, j) in zip(data, jekpglib.zz_scans[N]):
                block[i, j] = coeff
                if jekpglib.config.quant:
                    block[i, j] *= m_quant[i][j]

            if jekpglib.config.dct:
                m_dct = jekpglib.m_dcts[N]
                block = np.round_((m_dct.T).dot(block).dot(m_dct))

            self.channels[ch][x1:x2, y1:y2] = block

        if jekpglib.config.uv_subsample:
            self.co, self.cg = map(
                jekpglib.upsample2x, (self.co, self.cg))

        self.decoded_bgr = [
            jekpglib.ycocg2rgb(ycocg).clip(0, 255)[::-1]
                for ycocg in np.dstack((self.y, self.co, self.cg))
                    .reshape((self.w * self.h, 3))]

        out_fn = self.fn[:-6] + "_DEC.bmp"
        jekpglib.writebmp(  out_fn, 
                            self.decoded_bgr, 
                            self.w, 
                            self.h, 
                            self.outheader)

        return out_fn
