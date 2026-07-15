import functions as f
import random
import numpy as np
import cv2
from pathlib import Path

start = Path.cwd().parent
img_path = start / "images"
img_output_path = img_path / "output"
random.seed()

def rand_incr(img: np.ndarray) -> np.ndarray:
    h,w,_ = img.shape
    res = img.copy()
    rval = random.randint(1, 50)
    for i in range(0,rval):
        rcol = random.randint(0,h-1)
        rline = random.randint(0,w-1)
        res[rcol][rline] = 255
        #rcol = random.randint(0,h-1)
        #rline = random.randint(0,w-1)
        #res[rcol][rline] = 0
    
    #result = cv2.hconcat([img,res])
    #cv2.imshow(" ",result)
    #cv2.waitKey(0)
    return res

img = cv2.imread(img_path/"lena_color_512.tif")
crop_img = f.resize_img(img=img)
del img

quality = 90
tamper_img = f.create_tamper_evident(img=crop_img, quality=quality)
del crop_img
tamper_img = f.resize_img(img=tamper_img)
change_img = rand_incr(tamper_img)
mask, diff = f.check_tamper(img=change_img, quality=quality)
f.show_diff(img=tamper_img, mask=mask, outputPath=img_output_path/"lena_color_512")