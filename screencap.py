import cv2
import numpy as np
from PIL import ImageGrab

while True:
    img = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape
    print(height, width)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0Xff == ord('q'):
        break
    
cv2.destroyAllWindows()