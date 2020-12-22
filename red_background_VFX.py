import numpy as np
import cv2
import matplotlib.pyplot as plt

def nothing(x):
    pass
background = cv2.imread("balloons.jpg")
##background = cv2.cvtColor(background,cv2.COLOR_BGR2HSV)
cap = cv2.VideoCapture(0)

cv2.namedWindow('range')
cv2.createTrackbar('B1','range',0,255,nothing)
cv2.createTrackbar('G1','range',0,255,nothing)
cv2.createTrackbar('R1','range',0,255,nothing)
cv2.createTrackbar('B2','range',0,255,nothing)
cv2.createTrackbar('G2','range',0,255,nothing)
cv2.createTrackbar('R2','range',0,255,nothing)

while True:
    ret, vid = cap.read()
    ##vid = cv2.cvtColor(vid,cv2.COLOR_BGR2HSV)
    vid = np.flip(vid,axis=1)
    b1 = cv2.getTrackbarPos('B1','range')
    g1 = cv2.getTrackbarPos('G1','range')
    r1 = cv2.getTrackbarPos('R1','range')
    b2 = cv2.getTrackbarPos('B2','range')
    g2 = cv2.getTrackbarPos('G2','range')
    r2 = cv2.getTrackbarPos('R2','range')
    h,w = vid.shape[:2]
    background = cv2.resize(background,(w,h))
    
    ##vid = [h,s,v]
    lred = np.array([b1,g1,r1])
    ured = np.array([b2,g2,r2])
    mask1 = cv2.inRange(vid,lred,ured)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=2)
    mask1 = cv2.dilate(mask1,np.ones((3,3),np.uint8),iterations = 1)
    mask2 = cv2.bitwise_not(mask1)
    res1 = cv2.bitwise_and(background,background,mask=mask1)
    res2 = cv2.bitwise_and(vid,vid,mask=mask2)
    final = cv2.addWeighted(res1,1,res2,1,0)
    
    cv2.imshow('res1',res1)
    cv2.imshow('res2',res2)
    res = cv2.bitwise_and(vid,vid,mask=mask1)
    cv2.imshow('final',final)
    
    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
 
