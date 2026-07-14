import functions as f
import random
import numpy as np
from pathlib import Path

start = Path.cwd().parent
img_path = start / "images"
img_output_path = img_path / "output"
random.seed()

def rand_incr(img: np.ndarray) -> np.ndarray:
    h,w,_ = img.shape
    rand_cols = []
    rand_lines = []
    for i in range(0,20):
        rcol = random.randint(0,h-1)
        rline = random.randint(0,w-1)
        rand_cols.append(rcol)
        rand_lines.append(rline)