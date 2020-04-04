import cv2
import numpy as np

image_path1='ManimaranB.jpg'
image_path2='u2.jpg'
image1=cv2.imread(image_path1)#known
image2=cv2.imread(image_path2)#Unknown
faceclassifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face1=faceclassifier.detectMultiScale(image1,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
face2=faceclassifier.detectMultiScale(image2,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))#returns [[]]
for (x,y,w,h) in face1:
	i1=image1[y:y+h,x:x+w]
	cv2.imwrite('r1.jpg',i1)
for (x,y,w,h) in face2:
	i2=image2[y:y+h,x:x+w]
	cv2.imwrite('r2.jpg',i2)
res=cv2.matchTemplate(i1,i2,cv2.TM_CCOEFF_NORMED)
if len(res)==1 and res>=0.8:
	print(image_path1[0:-4])
else:
	print('Face not matched')