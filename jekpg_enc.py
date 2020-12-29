from jekEncDec import jekEncoder
import sys
if __name__ == "__main__":   
    if len(sys.argv) != 2:
        print("usage: py jekpg_enc.py <bmp fn>"); sys.exit()
    if not sys.argv[-1].endswith('.bmp'):
        print("Input is not a bmp!"); sys.exit()
    jekEncoder(sys.argv[-1]).encode()
