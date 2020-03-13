from PIL import Image
import numpy as np
import cv2
import sys

path = "./" + sys.argv[1]
binpath = './demo.bin'


image_orig = Image.open(path).convert('RGBA')
width = image_orig.size[0]
height = image_orig.size[1]


image_resize = image_orig.resize((width, height))
np.array(image_resize).tofile(binpath)

print("width:",width, " height:",height)
print("done")
