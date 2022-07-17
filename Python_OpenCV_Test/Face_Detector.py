import cv2


#opencvから顔認識するデータを読み込む
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#opencvから笑顔認識のデータを読み込む
smile_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

#写真を選択する
#img = cv2.imread('Tho.png')

#動画を選択（webcamの場合は '0')
webcam = cv2.VideoCapture(0)

#連続てにビデオのフレームから読み込む
while True:
    #webcamから抽出するフレームを読み込む
    successful_rate_read, frame = webcam.read()
    if not successful_rate_read:
        break

    #黒白写真に変換する
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #黒白写真の顔認識 
    face_detector = trained_face_data.detectMultiScale(grayscaled_img)

    #黒白写真の顔認識のデータを用いてフレームに方形式をつける
    for (x,y,w,h) in face_detector:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #写真の顔認識部分を抽出する
        face = frame[y:y+h ,x:x+w:]

        #顔認識部分を黒白写真に変換する
        grayscaled_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # haarcascadesの笑顔のデータを用いて、自分で考えて、顔の写真認識AIを入れる 
        #黒白写真の笑顔認識
        smile_detector = smile_data.detectMultiScale(grayscaled_face, scaleFactor=1.3, minNeighbors=17)
        
        #顔認識の方形式に笑顔認識の方形式につける
        for (x1,y1,w1,h1) in smile_detector:
            cv2.rectangle(face, (x1, y1), (x1+w1, y1+h1), (20, 52, 164), 4)

        #笑顔認識すれば、'Smile' という文字を下に貼る
        if len(smile_detector) > 0:
            cv2.putText(frame, 'Smile', (x+45, y+h+50), fontScale = 2,
            fontFace = cv2.FONT_HERSHEY_TRIPLEX, color = (220,20,60))

    #フレームを出す
    cv2.imshow('Face Detector',frame)

    #１ミリ/sの後で次のフレームが入る。連続的にビデオのように見える
    key = cv2.waitKey(1)
    
    #'q'と'Q'のキーを押すとフレームを閉じる
    if key == 81 or key == 113:
        break 

webcam.realease()
print('終了')
cv2.destroyAllWindows




"""
#顔認識
face_detector = trained_face_data.detectMultiScale(grayscaled_img)

#方形式で顔を選択する
#(x, y, w, h) = face_detector[2]
for (x,y,w,h) in face_detector:
    cv2.rectangle(webcam, (x, y), (x+w, y+h), (0, 255, 0), 2)


cv2.imshow('Face Detector',webcam)
cv2.waitKey(0)
"""


print('Hello')