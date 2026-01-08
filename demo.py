import cv2
import time 
from collections import Counter 
import numpy as np
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    if not ret:
        print(ret)
    # face detection with haarcascade 
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces on camera 
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    #process each face found + draw rectangle around it
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x + w, y + h+10), (0, 255, 0), 4)
        roi_gray_scale = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(frame[y:y + h, x:x + w], (224, 224)), -1), 0)

        # predict emotion on the cropped image
        # model prediction code comes here
        #placeholder until model is done ---------
        predicted_emotion = "happy"  # placeholder
        cv2.putText(frame, predicted_emotion, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        #----------------------------------------

    cv2.imshow('emotion detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

cap.release()
cv2.destroyAllWindows()



