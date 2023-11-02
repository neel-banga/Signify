import cv2
import streamlit as st
import numpy as np
from PIL import Image
import model
from playsound import playsound
from gtts import gTTS


def remove_adjacent_duplicates(input_str):
    if len(input_str) <= 1:
        return input_str

    result = [input_str[0]]

    for i in range(1, len(input_str)):
        if input_str[i] != input_str[i - 1]:
            result.append(input_str[i])

    return ''.join(result)

def say(words):
    tts = gTTS(text=words, lang='en-au', slow=False)
    tts.save("output.mp3")
    playsound("output.mp3")

def main_loop_1():
    sentance = ''
    st.title("Signify!")
    st.subheader("This app allows non-verbal people to communicate with others!")
    st.text("We show the text needed to communicate as well as saying it out loud")

    image_slot = st.empty()

    while True:
        frame, res = model.det1()
        image_slot.image(frame)

        boxes = res[0].boxes
        
        if len(boxes) == 0:
            try:
                say(sentance)
                sentance = ''
            except:
                pass

        try:
            box = boxes[0]
            label = int(box.cls)
            conf = float(box.conf)

            alpha = {}
            current_number = 0

            if conf >= 0.25:
                sentance += label
        except:
            pass

def main_loop_2():
    
    sentance = ''
    st.title("Signify!")
    st.subheader("This app allows non-verbal people to communicate with others!")
    st.text("We show the text needed to communicate as well as saying it out loud")

    image_slot = st.empty()
    obj_class = st.empty()

    while True:
        res = model.det2()
        frame = Image.open('prediction.jpg')
        image_slot.image(frame)

        #print(res['class'])
        for i in res['predictions']:
            print(i)
            _class = i['class']
            _conf = i['confidence']

            if _conf >= 0.25:
                sentance += _class.lower()

            try:
                #obj_class.text(_class)
                obj_class.text("Hi I am neel")
            except:
                pass

        else:
            try:
                say(remove_adjacent_duplicates(sentance))
                sentance = ''
            except:
                pass

if __name__ == '__main__':
    main_loop_1()
