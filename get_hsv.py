import cv2
import numpy as np

def nothing(x):
    pass

# 读取你的测试截图
image = cv2.imread('./test.png')
# 建议先裁剪出顶部进度条区域，方便观察
# image = image[50:150, 600:1300] 

cv2.namedWindow('HSV Tuner')

# 创建滑动条 (H: 0-179, S: 0-255, V: 0-255)
cv2.createTrackbar('HMin', 'HSV Tuner', 0, 179, nothing)
cv2.createTrackbar('SMin', 'HSV Tuner', 0, 255, nothing)
cv2.createTrackbar('VMin', 'HSV Tuner', 0, 255, nothing)
cv2.createTrackbar('HMax', 'HSV Tuner', 179, 179, nothing)
cv2.createTrackbar('SMax', 'HSV Tuner', 255, 255, nothing)
cv2.createTrackbar('VMax', 'HSV Tuner', 255, 255, nothing)

# 设置初始默认值 (比如先随便给个绿色的范围)
cv2.setTrackbarPos('HMin', 'HSV Tuner', 35)
cv2.setTrackbarPos('HMax', 'HSV Tuner', 85)
cv2.setTrackbarPos('SMin', 'HSV Tuner', 50)
cv2.setTrackbarPos('VMin', 'HSV Tuner', 50)

while True:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    h_min = cv2.getTrackbarPos('HMin', 'HSV Tuner')
    s_min = cv2.getTrackbarPos('SMin', 'HSV Tuner')
    v_min = cv2.getTrackbarPos('VMin', 'HSV Tuner')
    h_max = cv2.getTrackbarPos('HMax', 'HSV Tuner')
    s_max = cv2.getTrackbarPos('SMax', 'HSV Tuner')
    v_max = cv2.getTrackbarPos('VMax', 'HSV Tuner')
    
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    
    # 核心：根据阈值生成二值化掩码（白底黑字）
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    
    cv2.imshow('Original', image)
    cv2.imshow('Mask (White is target)', mask)
    cv2.imshow('Result', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # 按 Q 退出
        break

cv2.destroyAllWindows()
print(f"Lower Bound: np.array([{h_min}, {s_min}, {v_min}])")
print(f"Upper Bound: np.array([{h_max}, {s_max}, {v_max}])")