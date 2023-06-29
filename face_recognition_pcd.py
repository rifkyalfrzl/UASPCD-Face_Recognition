import tensorflow as tf
import cv2
import numpy as np

face_detect = cv2.CascadeClassifier('src/face.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

model = tf.keras.models.load_model('src/keras_model.h5')
class_names = ["Rifky Al Farezal", "Elon Musk"]

def get_class_name(class_no):
    if class_no < len(class_names):
        return class_names[class_no]
    else:
        return "Unknown"

while True:
    success, imgOrignal = cap.read()
    faces = face_detect.detectMultiScale(imgOrignal, 1.3, 5)
    
    for x, y, w, h in faces:
        crop_img = imgOrignal[y:y+h, x:x+h]
        img = cv2.resize(crop_img, (224, 224))
        img = tf.reshape(img, (1, 224, 224, 3))
        prediction = model.predict(img)
        classIndex = np.argmax(prediction)
        probabilityValue = np.amax(prediction)
        
        cv2.rectangle(imgOrignal, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(imgOrignal, (x, y-40), (x+w, y), (0, 255, 0), -2)
        cv2.putText(imgOrignal, str(get_class_name(classIndex)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue*100, 2)) + "%", (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Result", imgOrignal)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()