import sys
import cv2
import numpy as np

img = cv2.imread(sys.argv[1])
x1 = int(sys.argv[2])
y1 = int(sys.argv[3])
x2 = int(sys.argv[4])
y2 = int(sys.argv[5])

cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imwrite("final.jpg",img)
