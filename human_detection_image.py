import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import threading


def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        threading.Thread(target=process_image, args=(file_path,)).start()


def read_network_and_classes():
    net = cv2.dnn.readNetFromONNX("yolov5s.onnx") # Load the network from the ONNX file ( Open Neural Network Exchange)
    with open("objects.txt","r") as file:
        classes = file.read().split('\n')
    return net, classes

def get_detections(frame, net):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (640, 640), (0, 0, 0), True, crop=False) # Create a 4D blob from a frame
    net.setInput(blob)
    detections = net.forward()[0]
    return detections, width, height

def process_detections(detections, width, height): # Process the detections
    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    x_scale = width/640
    y_scale = height/641

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence >= 0.4:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] >= 0.25:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx- w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                wv= int(w * x_scale)
                hv = int(h * y_scale)
                box = np.array([x1,y1,wv,hv])
                boxes.append(box)
    return boxes, confidences, classes_ids

def draw_boxes(frame, boxes, confidences, classes_ids, classes):
    if len(boxes) > 0:
        num_retained_boxes = cv2.dnn.NMSBoxes(boxes,confidences,0.25,0.45)
        for i in num_retained_boxes:
            bbox = boxes[i]
            x1,y1,w,h = bbox
            text = classes[classes_ids[i]] + "{:.2f}".format(confidences[i])
            
           
            if confidences[i] > 0.8:
                color = (0, 255, 0)  
            elif 0.6 <= confidences[i] <= 0.8:
                color = (0, 255, 255)  
            else:
                color = (0, 0, 255) 

            cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),color,2)
            cv2.putText(frame, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,(255,0,255),2)
    return frame

def display_frame(frame):
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(file_path):
    net, classes = read_network_and_classes()
    frame = cv2.imread(file_path)
    detections, width, height = get_detections(frame, net)
    boxes, confidences, classes_ids = process_detections(detections, width, height)
    frame = draw_boxes(frame, boxes, confidences, classes_ids, classes)
    display_frame(frame)    
