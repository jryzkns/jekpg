from jekEncDec import jekDecoder
import sys
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: py jekpg_dec.py <jekpg fn>"); sys.exit()
    if not sys.argv[-1].endswith('.jekpg'):
        print("Input is not a jekpg!"); sys.exit()
    jekDecoder(sys.argv[-1]).decode()
