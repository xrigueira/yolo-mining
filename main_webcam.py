import cv2
import numpy as np
from turtle import width

# Pass the weights and the configurations
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Extract object names from the coco file
classes = []
with open('coco.names', 'r') as f:
    
    classes = f.read().splitlines()

# Load target video
cap = cv2.VideoCapture(0)

while cap.isOpened():

    _, img = cap.read()
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
            
            if confidence > 0.2:
                
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    # Get ride of of redundant boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    indexes = np.array(indexes)

    # Display results
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = np.random.uniform(0, 255, size=(len(boxes), 3)) # assign random colors to all baxes (3 channels)
   
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i]))
        color = colors[i]
        
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        (w_text, h_text), _ = cv2.getTextSize(label + ' ' + confidence, font, 0.8, 2)
        cv2.rectangle(img, (x, y), (x+w_text, y-h_text-8), color, -1)
        cv2.putText(img, label + ' ' + confidence, (x, y-5), font, 0.8, (255, 255, 255), 2)

    # To only keep the maximum score box

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)

    if key==27:
        break

cap.release()
cv2.destroyAllWindows()