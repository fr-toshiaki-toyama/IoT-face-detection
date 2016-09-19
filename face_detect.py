"""Facial detection using opencv and video capture"""
import time
import cv2
import pygame

DATA_PATH = './data/'
FACE_CASCADE_PATH = DATA_PATH + 'haar_face.xml'
EYE_CASCADE_PATH = DATA_PATH + 'haar_eye.xml'
FACE_CASCADE = cv2.CascadeClassifier(FACE_CASCADE_PATH)
EYE_CASCADE = cv2.CascadeClassifier(EYE_CASCADE_PATH)
SUCCESS_SOUND = DATA_PATH + 'hello.wav'
INTERVAL_IN_SECS = 0.2

def start_webcam(mirror=False):
    """Start capture on webcam"""
    cam = cv2.VideoCapture(0)
    # set framerate
    cam.set(cv2.cv.CV_CAP_PROP_FPS, 0.1)

    while True:
        _, img = cam.read()
        img = detect_faces(img)
        if mirror:
            img = cv2.flip(img, 1)
            cv2.imshow('Detection mode', img)
        if cv2.waitKey(1) == 27:
            break
        time.sleep(INTERVAL_IN_SECS)
    cv2.destroyAllWindows()

def detect_faces(image):
    """Facial feature detection per frame"""
    # RGB to gray bands
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    print "Detected {0} faces".format(len(faces))
    if len(faces) > 0:
        pygame.mixer.init()
        pygame.mixer.music.load(SUCCESS_SOUND)
        pygame.mixer.music.play()

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Image subset
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        # Detect and draw rectangle around eyes
        eyes = EYE_CASCADE.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return image

start_webcam(mirror=True)
