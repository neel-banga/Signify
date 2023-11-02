from flask import Flask, render_template, request, flash
import os
import model
import cv2
import shutil

try: 
   shutil.rmtree('frames')
except:
   print('DOES NOT EXIST')

app = Flask(__name__)

@app.route('/')
def indx():
  return render_template('index.html')

@app.route('/camera')
def camera():
   return render_template('camera.html')


@app.route('/upload_photo', methods=['POST'])
def upload_photo():
  z = ''
  sentance = ''
  video = request.files['photo']
  video_path = os.path.join(os.getcwd(), video.filename)

  video.save(video_path)

  output_folder = 'frames'

  if not os.path.exists(output_folder):
      os.makedirs(output_folder)

  cap = cv2.VideoCapture(video_path)
  fps = int(cap.get(cv2.CAP_PROP_FPS))


  frame_count = 0
  x = 0
  while True:
      ret, frame = cap.read()

      if not ret:
          break
      
      if x % fps == 0:
        print(f'GANG : {x}')
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)

      frame_count += fps
      x += 1

  cap.release()

  sentance = ''

  for f in os.listdir('frames'):
      z = model.get_letter_from_path(os.path.join('frames', f))
      if z:
         sentance += z
      print(z)
  print(sentance)

  return render_template('camera.html', text = sentance)


app.run(host='0.0.0.0', port=18)