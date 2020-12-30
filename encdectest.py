from jekEncDec import *
import os, sys

if __name__ == "__main__":
    enc_fn = jekEncoder(sys.argv[1]).encode()
    print(f"enc complete: {enc_fn}")
    dec_fn = jekDecoder(enc_fn).decode()
    print(f"dec complete: {dec_fn}")
