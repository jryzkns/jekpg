import os

for f__ in os.listdir("data"):
    if f__.endswith("_DEC.bmp") or f__.endswith(".jekpg"):
        print(f"Deleting:\t{f__}")
        os.remove(os.path.join("data", f__))