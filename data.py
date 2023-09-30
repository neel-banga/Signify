import os
import shutil
from PIL import Image, ImageFilter
import numpy as np
from random import randint


def image_to_folder():
   for f in os.listdir():
       if f.endswith('.png'):
           name = f.replace('.png', '')
           os.mkdir(name)
           shutil.move(f, os.path.join(os.getcwd(), name, f))


def create_blurred_images():
   for fd in os.listdir():
       if len(fd) == 1:
           for f in fd:
               img = Image.open(os.path.join(fd, f+'.png'))
               blurred_img = img.filter(ImageFilter.GaussianBlur(randint(10, 20)))               
               letter = f.replace('.png', '')
               blurred_img.save(os.path.join(os.getcwd(), fd, f'{letter}_blur.png'))


def create_rotated_images():
   for fd in os.listdir():
       if len(fd) == 1:
           for f in fd:
               img = Image.open(os.path.join(fd, f+'.png'))
               rotation_img = img.rotate(randint(0, 45))               
               letter = f.replace('.png', '')
               rotation_img.save(os.path.join(os.getcwd(), fd, f'{letter}_rotation.png'))


def combine_folders():
   os.mkdir('LETTERS')
   for fd in os.listdir():
       if len(fd) == 1:
           shutil.move(fd, 'LETTERS')


def create_new_data():
   remove_files(blur = True, rotation = True, inverted = True)
   create_blurred_images()
   create_rotated_images()


def remove_files(blur = False, rotation = False, original = False):
   for fd in os.listdir('LETTERS'):
       if len(fd) == 1:
           for f in os.listdir(os.path.join('LETTERS', fd)):       
               letter = fd
               try:
                   if blur: os.remove(os.path.join(os.getcwd(), 'LETTERS', fd, f'{letter}_blur.png'))
               except FileNotFoundError:
                   print('Image does not exist!')
              
               try:
                   if rotation: os.remove(os.path.join(os.getcwd(), 'LETTERS', fd, f'{letter}_rotation.png'))
               except FileNotFoundError:
                   print('Image does not exist!')
               try:
                   if original: os.remove(os.path.join(os.getcwd(), 'LETTERS', fd ,f'{letter}.png'))
               except FileNotFoundError:
                   print('Image does not exist!')
'''
def remove_files(blur = False, rotation = False, original = False):
   for fd in os.listdir('LETTERS'):
       if len(fd) == 1:
           for f in os.listdir(os.path.join('LETTERS', fd)):       
               letter = fd
               if blur: os.remove(os.path.join(os.getcwd(),'LETTERS', fd, f'{letter}_blur.png'))               
               if rotation: os.remove(os.path.join(os.getcwd(),'LETTERS', fd, f'{letter}_rotation.png'))
               if original: os.remove(os.path.join(os.getcwd(), 'LETTERS', fd, f'{letter}.png'))
'''
remove_files(blur = True, rotation = True)