import cv2
import numpy as np

# Haar Cascade 파일 경로 확인
print("Haar Cascade 경로:", cv2.data.haarcascades)

# OpenCV 테스트
img = np.zeros((100, 100), dtype=np.uint8)
cv2.imshow("Test Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
