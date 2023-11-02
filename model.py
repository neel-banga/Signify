from ultralytics import YOLO
from roboflow import Roboflow
import torch
import cv2
from torchvision import transforms
from PIL import Image
import os
import time
import numpy as np

rf = Roboflow(api_key="GjOibvXZTUUkYq5FHPvo")
project = rf.workspace().project("american-sign-language-letters")

def download_dataset():
    rf = Roboflow(api_key="GjOibvXZTUUkYq5FHPvo")
    project = rf.workspace("neel-hjdrl").project("sing-language-vghln")
    dataset = project.version(1).download("yolov8")
    return dataset

def train_model():
    dataset = download_dataset()

    model = 'yolov8n.pt'
    epochs = 25
    imgsz = 640
    batch = 8
    data = f'{dataset.location}/data.yaml'

    model = YOLO(model)

    results = model.train(data=data, epochs=5)
    results = model.val()

    return model

img_transforms = transforms.Compose([
   transforms.Resize((600, 600)),
   transforms.Grayscale()
])


def get_letter_from_path(model_path, path):

  model = YOLO(model_path)

  img = Image.open(path)
  img = img.resize((640, 640))
  img = img.convert('L')
  
  print(model.predict(img))  

  return model.predict(img)

def get_letter(model_path, img):

  model = YOLO(model_path)

  img = img.resize((600, 600))
  img = img.convert('L')

  return model.predict(img)
'''
results = get_letter_from_path('best.pt', 'Sing-Language-1/test/images/R18_jpg.rf.ee82b380feab18cc5c6a31a956dac18a.jpg')


annotated_frame = results[0].plot()
cv2.imshow('yay', annotated_frame)
cv2.waitKey(0)


boxes = results[0].boxes
box = boxes[0]
label = int(box.cls)
confidence = float(box.conf)

print(label, confidence)

alpha = {}
current_number = 0

for char in 'abcdefghijklmnopqrstuvwxyz':
    alpha[current_number] = char
    current_number += 1

letter = alpha[label]
print(letter)
'''
def load_yolo_model(model_path='model_files/runs/detect/train2/weights/last.pt'):
    model = YOLO(model_path)
    return model

def detect_sign_language(model, frame):
    # Resize and convert the frame to the appropriate format
    img = Image.fromarray(frame)
    img = img.resize((640, 640))
    img = img.convert('L')

    # Perform object detection using YOLO
    results = model.predict(img)

    return results

def det1():
    model = load_yolo_model('/Users/neelbanga/Documents/Coding/CongressionalApp/model_files/runs/detect/train2/weights/best.pt')

    # Open a connection to the webcam (usually camera index 0)
    cap = cv2.VideoCapture(0)
    cv2.resizeWindow("window", 640, 640) 

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        time.sleep(2)

        if not ret:
            break

        # Detect sign language in the frame
        results = detect_sign_language(model, frame)

        # Draw bounding boxes and labels on the frame
        for result in results:
            annotated_frame = result.plot()
            frame = np.array(annotated_frame)

        return frame, results

        # Display the frame with detections
        cv2.imshow('Sign Language Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


model = project.version(6).model
cap = cv2.VideoCapture(0)
def det2():

    ret, frame = cap.read()

    if not ret:
        return

    cv2.imwrite('img.jpg', frame) 
    # Detect sign language in the frame
    pred = model.predict("img.jpg", confidence=40, overlap=30)
    results = pred.json()
    pred.save("prediction.jpg")
    print(results)
    return results
    
def get_letter_from_path(path):

    model = project.version(6).model
    results = model.predict(path, confidence=40, overlap=30).json()
    model.predict(path, confidence=40, overlap=30).save("prediction.jpg")

    for i in results['predictions']:
        _class = i['class']
        _conf = i['confidence']

        return _class


def detection():

    model = project.version(6).model

    # Open a connection to the webcam (usually camera index 0)
    cap = cv2.VideoCapture(0)
    cv2.resizeWindow("window", 640, 640) 

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        #time.sleep(2)

        if not ret:
            break

        cv2.imwrite('img.jpg', frame) 
        # Detect sign language in the frame
        results = model.predict("img.jpg", confidence=40, overlap=30).json()
        model.predict("img.jpg", confidence=40, overlap=30).save("prediction.jpg")
        frame = Image.open('prediction.jpg')

        cv2.imshow('Sign Language Detection', frame)