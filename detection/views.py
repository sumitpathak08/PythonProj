from django.shortcuts import render,redirect
from django.http import HttpResponse

import cv2
import numpy as np
from face_detection import settings
# Create your views here.

def index(request):
    return render(request, 'index.html')

def home(request):
    return HttpResponse('Hello Dear Friend')

def detect(request):
    face_cascade = cv2.CascadeClassifier(str(settings.BASE_DIR) + '/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Video")
    while True:
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 2)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('Video',img)

        k = cv2.waitKey(30) & 0xff
        if k==27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect('/')

