import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
from gtts import gTTS
from playsound import playsound
import time
import cv2

def say(words):
    tts = gTTS(text=words, lang='en-au', slow=False)
    tts.save("output.mp3")
    playsound("output.mp3")


alpha = {}
current_number = 0

# - = del
# ! = nothing
# + = space
for char in 'abcdefghijklmnopqrstuvwxyz-!+':
    alpha[current_number] = char
    current_number += 1


img_transforms = transforms.Compose([
   transforms.Resize((50, 50)),
   transforms.Grayscale(),
   transforms.ToTensor()
])

training = datasets.ImageFolder('ASL_Alphabet_Dataset/asl_alphabet_train', transform=img_transforms)
train_loader = torch.utils.data.DataLoader(training, batch_size=10, shuffle=True)

class Model(nn.Module):
   def __init__(self):
       super().__init__()

       self.convolution = nn.Sequential(
           nn.Conv2d(1, 64, (5, 5)),
           nn.ReLU(),
           nn.Conv2d(64, 64, (5, 5)),
           nn.ReLU(),
           nn.Conv2d(64, 64, (5, 5)),
           nn.ReLU(),
           nn.Flatten(),
           nn.Linear(92416, 30)
       )

   def forward(self, x):
       return self.convolution(x)
  
def train_model():
   model = Model()
   opt = optim.Adam(model.parameters(), lr = 0.00001)
   loss_fn = nn.CrossEntropyLoss()

   for epoch in range(5):
       for batch in train_loader:
           X, y = batch
           output = model(X)          
           loss = loss_fn(output, y)
           opt.zero_grad()
           loss.backward()
           opt.step()

       print(f'LOSS: {loss}')
       torch.save(model.state_dict(), 'model.pth')

#train_model()
'''
model = Model()
model.load_state_dict(torch.load('model.pth'))

print(alpha)

img = Image.open('ASL_Alphabet_Dataset/asl_alphabet_train/nothing/nothing3.jpg')
img_transformed = img_transforms(img)
img_batch = img_transformed.unsqueeze(0)
output = model(img_batch)
predicted_class = torch.argmax(output)
print(predicted_class.item())

letter = alpha[predicted_class.item()]
print(letter)

base_dir = 'ASL_Alphabet_Dataset/asl_alphabet_test'
for i in os.listdir(base_dir):
    
    img = Image.open(os.path.join(base_dir, i))
    img_transformed = img_transforms(img)
    img_batch = img_transformed.unsqueeze(0)
    output = model(img_batch)
    predicted_class = torch.argmax(output)

    letter = alpha[predicted_class.item()]
    ans = i.replace('_test.jpg', '')
    print(f'GUESS: {letter}, ANSWER: {ans}')
'''

def get_sentance():
    sentance = ''
    model = Model()
    model.load_state_dict(torch.load('model.pth'))
    start_time = time.time()
    cap = cv2.VideoCapture(0)
    while time.time() - start_time < 25:
        for i in range(5):
            print(i)
            time.sleep(1)
        ret, frame = cap.read()

        cv2.imshow('Frame', frame) 
        cv2.imwrite('frame.jpg', frame)
        

        img = Image.open('frame.jpg')
        img_transformed = img_transforms(img)
        img_batch = img_transformed.unsqueeze(0)
        output = model(img_batch)
        predicted_class = torch.argmax(output)
        letter = alpha[predicted_class.item()]
        print(letter)
        sentance += letter
    
    return sentance

s = get_sentance()
print(s)
say(s)