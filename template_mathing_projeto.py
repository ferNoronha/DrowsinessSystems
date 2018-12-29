import cv2
import cv2 as conv2
import numpy as np
import matplotlib.pyplot as plt

threshold = 0.94
ex = int(0)
ey = int(0)
ew = int(0)
eh = int(0)
ex2 = int(0)
ey2 = int(0)
eh2 = int(0)
ew2 = int(0)


def validar_camera(mirror, img):
    if mirror:
        img = cv2.flip(img, 1)
    return img


template = cv2.imread('C:/Users/Faculdade/Desktop/python/projetos python/olhoteste.jpg', 1)

face_cascata = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
#olho_cascata = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_eye.xml')
olho_cascata = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
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
        # print(faces)
        for (x, y, w, h) in faces:
            img_face = img[y:y + h, x:x + w]
            if img_face is not None:
                img_face_original = img_face.copy()
            olhos = olho_cascata.detectMultiScale(img_face)
            centroX = (w - x)
            # print(olhos)
            # print(centroX)
            centroOlho = (float)(0)
            # cv2.waitKey(0)
            if olhos is not None and olhos != [] and len(olhos) >= 2:
                if olhos[0] is not None and olhos[0] != []:
                    # print ('olho[1]: '+ str(olhos[0]))
                    # cv2.waitKey(0)
                    ex, ey, ew, eh = olhos[0]
                if olhos[1] is not None and olhos[1] != []:
                    # print ('olho[1]: '+str(olhos[1]))
                    # cv2.waitKey(0)
                    ex2, ey2, ew2, eh2 = olhos[1]
                if ex < ex2:
                    img_olho = img_face[ey:ey + eh, ex:ex + ew]
                    img_olhod = img_face[ey2:ey2 + eh2, ex2:ex2 + ew2]
                else:
                    img_olhod = img_face[ey:ey + eh, ex:ex + ew]
                    img_olho = img_face[ey2:ey2 + eh2, ex2:ex2 + ew2]

                # img_olho.astype(np.uint8)
                # template.astype(np.uint8)
                # print(img_olho)
                #ret3, th3 = cv2.threshold(img_olho, 127, 255, cv2.THRESH_BINARY)
                #ret4, templ = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)

                res = cv2.matchTemplate(img_olho, template, cv2.TM_CCORR_NORMED)
                # ret3,th3 = cv2.threshold(res,0.85,255,cv2.THRESH_BINARY)
                # print(res)
                if res is not None:
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    # cv2.imshow('thresold',th3)
                    # bottom = (max_loc[0] + ww, max_loc[1] + hh)
                    # cv2.rectangle(img_olho,max_loc, bottom_right, 255, 2)
                    print(max_val)
                    print(min_val)
                    cv2.putText(img_face_original, "Max: "+(str)(max_val), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(img_face_original, "Min: "+(str)(min_val), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(img_face_original, "Threshold: " +(str)(threshold), (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
                    if min_val > threshold:
                        cv2.putText(img_face_original, "Aberto", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
                    else:
                        cv2.putText(img_face_original, "Fechado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
                    cv2.imshow('result',res)

        if img_face is not None:
            cv2.imshow('face', img_face_original)
        if img_olho is not None:
            cv2.imshow('olho', img_olho)
        if img_olhod is not None:
            cv2.imshow('olhod', img_olhod)
        cv2.imshow('template',res)
        cv2.imshow('template',template)

        key = cv2.waitKey(1)

        if key == ord('p'):
            cv2.imwrite('olho.jpg', img_olho)
        if key == ord('f'):
            cv2.imwrite('face.jpg', img_face)
        if key == ord('q'):
            break;

cap.release()
cv2.destroyAllWindows()
