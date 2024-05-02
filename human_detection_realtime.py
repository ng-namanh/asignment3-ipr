import cv2
import numpy as np

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print(cv2.__version__)
            
cap = cv2.VideoCapture(0) ## Capture video from camera

net = cv2.dnn.readNetFromONNX("yolov5s.onnx")
file = open("objects.txt","r") 
classes = file.read().split('\n')

def get_detections(frame, net):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (640, 640), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward()[0]
    return detections, width, height

def process_detections(detections, width, height):
    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    x_scale = width/640
    y_scale = height/640

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


def calculate_and_display_fps(frame, timer):
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, "FPS : " + str(int(fps)), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    return frame

def display_frame(frame):
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        return False
    return True

while True:
    ok, frame = cap.read()
    if not ok:
        break

    timer = cv2.getTickCount()
    detections, width, height = get_detections(frame, net)
    boxes, confidences, classes_ids = process_detections(detections, width, height)
    frame = draw_boxes(frame, boxes, confidences, classes_ids, classes)
    frame = calculate_and_display_fps(frame, timer)
    if not display_frame(frame):
        break

cap.release()