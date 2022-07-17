import os
import re
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
print(tf.__version__)


#表現の訓練したデータを読み込む
model = load_model("best_model.h5")

#opencvから顔認識するデータを読み込む
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#動画を選択（webcamの場合は '0')
webcam = cv2.VideoCapture(0)

#連続てにビデオのフレームから読み込む
while True:
    #webcamから抽出するフレームを読み込む
    successful_rate_read, frame = webcam.read()  
    if not successful_rate_read:
        break
    
    #RGB写真に変換する
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #黒白写真の顔認識
    faces_detected = trained_face_data.detectMultiScale(gray_img, 1.32, 5)

    #黒白写真の顔認識のデータを用いてフレームに方形式をつける
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #写真の顔認識部分を抽出する
        face = gray_img[y:y + w, x:x + h]  

        #顔の写真を調整する
        face = cv2.resize(face, (224, 224))

        #Kerasで写真を配列に変換する
        img_pixels = image.img_to_array(face)

        #ピクセルに分ける
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        #データを訓練する
        predictions = model.predict(img_pixels)

        #配列の長さを探す
        max_index = np.argmax(predictions[0])

        #表現を定義する
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        #表現の文字を下に貼る
        cv2.putText(frame, predicted_emotion, (x+45, y+h+50), fontScale = 2,
        fontFace=cv2.FONT_HERSHEY_TRIPLEX, color = (0, 0, 255))

    #フレームを出す
    cv2.imshow('Facial_Emotion_Detector', frame)

    #１ミリ/sの後で次のフレームが入る。連続的にビデオのように見える
    key = cv2.waitKey(1)

    #'q'と'Q'のキーを押すとフレームを閉じる
    if key == 81 or key  == 113:
        break

webcam.release()
cv2.destroyAllWindows