import cv2
import datetime

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

while True:
    _ ,Frame = cap.read()
    OriginalFrame = Frame.copy()
    Gray = cv2.cvtColor(Frame,cv2.COLOR_BGR2GRAY)

    face = faceCascade.detectMultiScale(Gray, 1.3, 6)
    for x, y, w, h in face:
        cv2.rectangle(Frame,(x,y),(x+w,y+h),(0,255,255),3)
        face_region = Frame[y:y+h, x:x+w]
        gray_region = Gray[y:y+h, x:x+w]
        smile = smileCascade.detectMultiScale(gray_region, 1.3, 30)
        for x1, y1, w1, h1 in smile:
            cv2.rectangle(face_region,(x1,y1),((x1+w1),(y1+h1)),(0, 0, 255), 3)
            TimeStamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            FileName = f'SelfieBaz{TimeStamp}.png'
            cv2.imwrite(FileName, OriginalFrame)

    cv2.imshow('Pakna', Frame)
    if cv2.waitKey(10) == ord('q'):
        break
