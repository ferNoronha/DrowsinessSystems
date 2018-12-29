import cv2
import time
import numpy as np
import winsound
from threading import Thread
import matplotlib.pyplot as plt

frequencia = 2500
duration = 100

def apita():
    winsound.Beep(frequencia, duration)

ex = int(0)
ey = int(0)
ew = int(0)
eh = int(0)
ex2 = int(0)
ey2 = int(0)
eh2 = int(0)
ew2 = int(0)

inicio = 0
fim = 0
cont = 0
t = 0
th = Thread(target=apita)

def validar_camera(mirror, img):
    if mirror:
        img = cv2.flip(img, 1)
    return img


face_cascata = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
#olho_cascata = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_eye.xml')
#olho_cascata = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
#olho_cascata = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
olho_cascata = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('teste.mp4')

while (True):
    # captura frame por frame
    ok, frame = cap.read(5)
    if ok == False:
        break
    else:
        img = cv2.flip(frame, 1)
        img_original = img.copy()
        cv2.imshow('teste', img_original)
        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascata.detectMultiScale(img_cinza, 1.3, 5)
        centroX = (float)(0)

        for (x, y, w, h) in faces:
            img_face = img[y:y + h, x:x + w]
            if img_face is not None and img_face != []:
                img_face_origin = img_face.copy()
                img_face_original = img[y:y+h-100, x:x +w]
                olhos = olho_cascata.detectMultiScale(img_face)
                centroX = (w - x)


                centroOlho = (float)(0)

                if olhos is not None and olhos != [] and len(olhos) >= 2:
                    if olhos[0] is not None and olhos[0] != []:
                        ex, ey, ew, eh = olhos[0]
                    if olhos[1] is not None and olhos[1] != []:
                        ex2, ey2, ew2, eh2 = olhos[1]
                if ex < ex2:
                    img_olho = img_face[ey+20:ey + eh, ex:ex + ew]
                    img_olhod = img_face[ey2+20:ey2 + eh2, ex2:ex2 + ew2]
                    img_cinza_olho = cv2.cvtColor(img_olho.copy(), cv2.COLOR_BGR2GRAY)
                else:
                    img_olhod = img_face[ey+20:ey + eh, ex:ex + ew]
                    img_olho = img_face[ey2+20:ey2 + eh2, ex2:ex2 + ew2]
                    img_cinza_olho = cv2.cvtColor(img_olho.copy(),cv2.COLOR_BGR2GRAY)

                if img_cinza_olho is not None:
                    im = cv2.GaussianBlur(img_cinza_olho, (3, 3), 0)



                    ret3, th3 = cv2.threshold(im, 50, 255, cv2.THRESH_BINARY_INV)
                    cv2.imshow('threshold',th3)
                    #mascara = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
                    mascara = np.ones((5, 5), np.uint8)
                    erosao = cv2.erode(th3, mascara, iterations=1)
                    cv2.imshow('erosao', erosao)
                    #masc = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
                    masc = np.ones((3, 3), np.uint8) # 5,5
                    dilatacao = cv2.dilate(erosao, masc)
                    cann = cv2.Canny(dilatacao, 5, 30)
                    cv2.imshow('dilatacao', dilatacao)

                    cv2.imshow('canny', cann)
                    cv2.imwrite('C:/Users/Faculdade/PycharmProjects/untitled/imagens/canny.jpg', cann)
                    circles = cv2.HoughCircles(cann, cv2.HOUGH_GRADIENT, 1, 4, np.array([]), 100, 10, 1, 20) #4 por 5
                    if circles is not None and circles != []:
                        a, b, c = circles.shape

                        cv2.putText(img_face_original, "Aberto", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),2)
                        if t==1:
                            inicio = 0
                            fim = 0
                            t = 0

                        for i in range(b):
                            cv2.circle(cann, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3,
                                       cv2.LINE_AA)
                    else:
                        if t==0:
                            inicio = time.time()
                            t = 1
                            cont = cont + 1
                        cv2.putText(img_face_original, "Fechado", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),2)
                    if t!=0:
                        fim = time.time()
                        sec = fim-inicio
                    else:
                        sec = 0

                    #if sec>=3:
                        #winsound.Beep(frequencia, duration)

                    cv2.putText(img_face_original, "" + (str)(cont), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
                    cv2.putText(img_face_original, "" + (str)(sec), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)

                if img_face is not None and img_face != []:
                    cv2.imshow('face', img_face_original)

                if img_olho is not None and img_olho!=[]:
                    cv2.imshow('olho', img_olho)



        key = cv2.waitKey(1)
        #cv2.waitKey(0)
        if key == ord('p'):
            cv2.imwrite('C:/Users/Faculdade/PycharmProjects/untitled/imagens/olho.jpg', img_olho)
            cv2.imwrite('C:/Users/Faculdade/PycharmProjects/untitled/imagens/dilatacao.jpg',erosao)
            cv2.imwrite('C:/Users/Faculdade/PycharmProjects/untitled/imagens/erosao.jpg', dilatacao)
            cv2.imwrite('C:/Users/Faculdade/PycharmProjects/untitled/imagens/threshold.jpg',th3)
            cv2.imwrite('C:/Users/Faculdade/PycharmProjects/untitled/imagens/canny.jpg',cann)
            cv2.imwrite('C:/Users/Faculdade/PycharmProjects/untitled/imagens/face.jpg',img_face)
            cv2.imwrite('C:/Users/Faculdade/PycharmProjects/untitled/imagens/recorte.jpg',img_face_original)
        if key == ord('f'):
            cv2.imwrite('face.jpg', img_face)
        if key == ord('q'):
            break;

cap.release()
cv2.destroyAllWindows()


