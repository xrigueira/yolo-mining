from turtle import width
import cv2
import numpy as np

# Pass the weights and the configurations
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Extract object names from the coco file
classes = []
with open('coco.names', 'r') as f:
    
    classes = f.read().splitlines()

# Load target image
img = cv2.imread('image.jpg')

height, width, _ = img.shape

# Rescale and normalize image
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

net.setInput(blob)

# Get the names of the output layers
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    
    for detection in output:
        
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.5:
            
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

print(len(boxes)) # does not detect anything. Fix before proceeding

# To only keep the maximum score box

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()