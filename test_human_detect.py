import cv2
from Human_Detection import Detector

img_path = 'test_images/test1.jpg'
img = cv2.imread(img_path)
if img is None:
    print('ERROR: could not load', img_path)
else:
    out = Detector(img)
    cv2.imwrite('out_detect.jpg', out)
    print('Wrote out_detect.jpg')
