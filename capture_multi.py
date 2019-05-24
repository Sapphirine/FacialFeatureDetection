import cv2
from fastai.vision import *
from fastai.vision.image import pil2tensor, Image

path = "C:\\Users\\Linsu Han\\Documents\\[COLUMBIA]\\EECS 6895 - Adv Big Data Analytics\\"
fc = cv2.CascadeClassifier(path + 'Final\\haarcascade_frontalface_default.xml')
learn = load_learner(path + 'Final\\models\\', 'attractive_male_resnet50.pkl')
category = 'Attractive Male'


def extractFaceCoords(img, fc, tolerance):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_coords = fc.detectMultiScale(gray, 1.2, tolerance, minSize=(60, 60))
    return face_coords


cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    # H, W, D = frame.shape
    face_coords = extractFaceCoords(frame, fc, 2)

    if face_coords is not None:
        for coords in face_coords:
            x, y, w, h = coords
            face_rgb = cv2.cvtColor(frame[y:y + h, x:x + h], cv2.COLOR_BGR2RGB)
            img_fastai = Image(pil2tensor(face_rgb, np.float32).div_(255))
            prediction = learn.predict(img_fastai)

            if int(prediction[0]) == 1:
                result = category + ': True'
            else:
                result = category + ': False'
            p = prediction[2].tolist()
            prob = 'Probability: ' + str(round(p[1], 3))
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 0), thickness=1)  # color in BGR
            cv2.putText(img=frame, text=prob, org=(x, y - 13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5,
                        color=(0, 255, 0), thickness=1)
            cv2.putText(img=frame, text=result, org=(x, y - 26), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5,
                        color=(0, 255, 0), thickness=1)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # quit
        break

    out.write(frame)
    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
