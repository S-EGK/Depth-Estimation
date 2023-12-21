import cv2


cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

num = 0


while cap.isOpened():
    succes1, img = cap.read()
    succes2, img2 = cap2.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        true1 = cv2.imwrite('sterro_vision/images/stereoLeft/imageL' + str(num) + '.png', img)
        true2 = cv2.imwrite('sterro_vision/images/stereoRight/imageR' + str(num) + '.png', img2)
        if true1 and true2:
            print("images saved!")
        num += 1

    cv2.imshow('Img 1',img)
    cv2.imshow('Img 2',img2)