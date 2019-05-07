
from __future__ import print_function
import numpy as np
import cv2
import sys
import keras
import csv
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import  os.path

import matplotlib.pyplot as plt




def get_mean_pixels(): #returns average face
    filename = "meanPixels.csv"
    fp = open(filename, "rU")
    csv_reader = csv.reader(fp)
    values = next(csv_reader)

    img = np.matrix(values)
    img = img.reshape(1, 48, 48, 1)
    img = img.astype(np.float)
    img = img.astype(int)
    # plt.imshow(img, cmap='gray', interpolation='nearest')
    # plt.show()
    return img


#takes in a grey face and a model and returns the correct index
def get_expression(img, model):

    # img = cv2.resize(img, (48, 48))
    # img = 255 - img 

    # img = cv2.equalizeHist(img)
    # plt.imshow(img, cmap='Greys')
    # plt.show()  

    
    img = np.array(img)

    # img = img/255


    img = img.reshape(1, 48, 48, 1)
    

    pred = model.predict(img, batch_size=1, verbose=0)
    score = np.max(pred)
    pred_label = np.argmax(pred[0])
    return pred_label




def run():
  angry = cv2.imread("angry.png", -1)
  alpha_angry = angry[:, :, 3] / 255.0

  happy = cv2.imread("happy.png", -1)
  alpha_happy = happy[:, :, 3] / 255.0

  neutral = cv2.imread("neutral.png", -1)
  alpha_neutral = neutral[:, :, 3] / 255.0

  disgust = cv2.imread("disgust.png", -1)
  alpha_disgust = disgust[:, :, 3] / 255.0

  sad = cv2.imread("sad.png", -1)
  alpha_sad = sad[:, :, 3] / 255.0

  suprised = cv2.imread("suprised.png", -1)
  alpha_suprised = suprised[:, :, 3] / 255.0


  afraid = cv2.imread("afraid.png", -1)
  alpha_afraid = afraid[:, :, 3] / 255.0


  model_filename = "model_4layer_2_2_pool.json"

  json_string = open(model_filename).read()
  model = model_from_json(json_string)
  model.summary()
  model.compile(loss=keras.losses.categorical_crossentropy,
          optimizer=keras.optimizers.Adadelta(),
          metrics=['accuracy'])

  weights_filename = "model_4layer_2_2_pool.h5"
  model.load_weights(weights_filename)

  faceCascade = cv2.CascadeClassifier( "haarcascade_frontalface_default.xml")
  cap = cv2.VideoCapture(0)

  mean_of_pixels = get_mean_pixels()

  while(1):
    i=0
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        x = (int)( x+ .1*w)
        y = (int) (y + .2*h)
        w = (int) (w*.8)
        h = (int) ( h*.8)
       

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)) 
        face = face - mean_of_pixels
        plt.imshow(face, cmap='gray', interpolation='nearest')
        plt.show()

        emotion = get_expression(face, model)


        # print(emotion)

        if (emotion ==0):
            # print("you are angry")
            emoji_face = angry
            alpha_face = alpha_angry

        elif (emotion ==1):
            # print("you are disgusted")
            emoji_face = disgust
            alpha_face = alpha_disgust



        elif (emotion ==2):
            # print("you are afraid")
            emoji_face = afraid
            alpha_face = alpha_afraid

        elif (emotion ==3):
            # print("you are happy")
            emoji_face = happy
            alpha_face = alpha_happy


        elif (emotion ==4):
            # print("you are sad")
            emoji_face = sad
            alpha_face = alpha_sad


        elif (emotion ==5):
            # print("you are suprised")
            emoji_face = suprised
            alpha_face = alpha_suprised


        else:
            # print("you are neutral")
            emoji_face = neutral
            alpha_face = alpha_neutral


        resized_emoji = cv2.resize(emoji_face, (h, w)) 
        alpha_emoji = resized_emoji[:, :, 3] / 255.0
        alpha_frame = 1.0 - alpha_emoji


        # frame[y:y+h,  x:x+w] = resized_emoition

        for c in range(0, 3):
          frame[y:y+h,  x:x+w, c] = (alpha_emoji * resized_emoji[:, :, c] + alpha_frame * frame[y:y+h,  x:x+w, c])
 
        cv2.imshow("Emoji Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()



if __name__ == "__main__":
  run()
  # get_mean_pixels()




