# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""


import cv2 as cv
import numpy as np
 
capture = cv.VideoCapture(0)

#arquivos necessarios para a deteccao de faces 
modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"
#carregamento da rede
net = cv.dnn.readNetFromTensorflow(modelFile, configFile)

#confiança minima para a deteccao da face
conf_threshold = 0.7
#se deu certo abrir a camera
if(capture.isOpened()):
    while(1):
        #pega frame atual
        ret, frame = capture.read()
        
        #print(frame.shape) #altura, largura, canais        
        im_height = frame.shape[0]
        im_width = frame.shape[1]
        
        #deteccao das faces
        #converte em blob
        #cv.resize(frame, (300, 300))
        blob = cv.dnn.blobFromImage(frame, 1.0, (300,300), [104, 117, 123], False, False)
        net.setInput(blob)
        #processa pela rede
        detections = net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                #os valores de saida da rede são normalizados entre [0,1] por isso a necessidade da multiplicacao
                x1 = int(detections[0, 0, i, 3] * im_width)
                y1 = int(detections[0, 0, i, 4] * im_height)
                x2 = int(detections[0, 0, i, 5] * im_width)
                y2 = int(detections[0, 0, i, 6] * im_height)
                
                #sub_face = frame[x1:x2, y1:y2]
                frame[y1:y2, x1:x2] = cv.blur(frame[y1:y2, x1:x2], (25,25))
                frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] - 10
                cv.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        
        #exibe a imagem processada 
        cv.imshow("Identity", frame)
       
        #interrompe o processo saindo do while
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
else:
    print('Erro ao abir a webcam')
 
#libera as extruturas e fecha as janelas
capture.release()
cv.destroyAllWindows()