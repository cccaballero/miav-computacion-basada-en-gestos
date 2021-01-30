#Inicialmente se importan los modulos a utilizar

import cv2
import numpy as np
import math

#Se inicializa la captura de video de Opencv
cap = cv2.VideoCapture(0)



def prepare_image(frame, roi):
    
    kernel = np.ones((3,3),np.uint8)
    
    #Se convierte el frame de video a HSV
    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    #Se define el rango de color de la piel
    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)
    
    #Se extrae la imagen basada en el color de la piel
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    #Se extrapola la mano para rellenar manchas oscuras
    mask = cv2.dilate(mask,kernel,iterations = 4)
    
    #Se desenfoca
    mask = cv2.GaussianBlur(mask,(5,5),100)

    return mask


def get_bigest_contour(mask):

    #Se encuentran los contornos 
    contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        #Devuelve el mayor contorno
        return max(contours, key = lambda x: cv2.contourArea(x))
    
    #Devuelve falso si no ecuentra algun contorno 
    return False


def get_defects(cnt):
    #Se aproxima la forma de contorno
    epsilon = 0.0005*cv2.arcLength(cnt,True)
    approx= cv2.approxPolyDP(cnt,epsilon,True)
    
    #Se obtiene el contorno alrrededor de la mano utilizando Convex Hull
    hull = cv2.convexHull(cnt)
    
    #Se definen las áreas
    areahull = cv2.contourArea(hull)
    areacnt = cv2.contourArea(cnt)
    
    #Se encuetra el porcentaje del área no cubierta por la mano
    arearatio = ((areahull - areacnt) / areacnt) * 100
    
    #Se encuentran los defectos en el Convex Hull con respecto a la mano
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)
    
    #Se define la variable `l` que contendrá el número de defectos
    l=0

    if defects is not None:
    
        #Se procede a encontrar el número de defectos en relación a los dedos
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100,180)
            
            #Se encuentra la longitud de todos los lados del triángulo 
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #Se encuentra la distancia entre el punto y el casco convexo
            d=(2*ar)/a
            
            #Se aplica la regla de coseno
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
            #Se ignoran los ángulos mayores a 90 grados y los puntos muy cercanos al casco convexo, ya que normalmente aparecen debido al ruido
            if angle <= 90 and d>30:
                l += 1
                cv2.circle(roi, far, 3, [255,0,0], -1)
            
            #Se dibujan lineas alrrededor de la mano
            cv2.line(roi,start, end, [0,255,0], 2)
            
        l+=1
    return l, areacnt, arearatio


def print_gestures(l, areacnt, arearatio):
    #Se mustran en pantalla los gestos que se encuantran en los rangos 
    font = cv2.FONT_HERSHEY_SIMPLEX
    if l==1:
        if areacnt < 2000:
            cv2.putText(frame,'Ponga la mano en el cuadro',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        else:
            if arearatio < 12:
                cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            elif arearatio < 17.5:
                cv2.putText(frame,'Pulgar arriba',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif l == 2:
        cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif l == 3:
            if arearatio < 27:
                cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame,'ok',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif l == 4:
        cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif l == 5:
        cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)


#Se inicia un ciclo infinito
while True:

    ret, frame = cap.read()
    frame=cv2.flip(frame,1)

    #Se define la región de interés
    roi=frame[100:300, 100:300]

    #se obtiene la mascara a partir de la obtención y preparación de la imagen
    mask = prepare_image(frame, roi)

    #Se obtiene el contorno mas grande
    cnt = get_bigest_contour(mask)

    if cnt is not False:
    
        #Se obtienen los defectos ademas del area del contorno y el porcentaje del área no cubierta por la mano
        l, areacnt, arearatio = get_defects(cnt)
        
        #Se identifican los gestos y se muestra la información visual
        print_gestures(l, areacnt, arearatio)
        
    #Se muestran las ventanas
    cv2.imshow('mask',mask)
    cv2.imshow('frame',frame)

    #Si se presiona la tecla "Esc" se termina el programa
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()

