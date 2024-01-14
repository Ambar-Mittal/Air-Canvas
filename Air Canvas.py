import cv2
import mediapipe as mp
import numpy as np
import os
import math
mp_hands = mp.solutions.hands

# webcam input
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 20) 
width = 1280
height = 720
cap.set(3, width) 
cap.set(4, height)

# Image that will contain the drawing and then passed to the camera image
imgCanvas = np.zeros((height, width, 3), np.uint8)

# Getting header images in a list
folderPath = '''C:\\Users\\ambar\\Desktop\\Python and Django Projects\\Air Canvas\\Header''' #replace this with the path from your original Header folder
myList = os.listdir(folderPath)

overlayList = []
for imagePath in myList:
    image = cv2.imread(f'{folderPath}/{imagePath}')
    overlayList.append(image)

# print(overlayList)
# Presettings:
header = overlayList[0]
drawColor = (0, 0, 255) #default colour
thickness = 20 # Thickness of the painting
tipIds = [4, 8, 12, 16, 20] # Fingertips indexes
xp, yp = [0, 0] # Coordinates that will keep track of the last position of the index finger

with mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.65, max_num_hands=1) as hands:
    while cap.isOpened():
        
        success, image = cap.read()
        
        if not success:
            # print("Ignoring empty camera frame.")
            break

        # Flip the image horizontally for selfie-view display
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = hands.process(image)
        # print(results)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            # print(results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                # Getting all hand points coordinates
                # print(hand_landmarks)
                points = []
                for lm in hand_landmarks.landmark:
                    # print(lm)
                    points.append([int(lm.x * width), int(lm.y * height)])
    
                # Only go through the code when a hand is detected
                # print(points)
                if len(points) != 0:
                    x1, y1 = points[8]  # Index finger
                    x2, y2 = points[12] # Middle finger
                    x3, y3 = points[4]  # Thumb
                    x4, y4 = points[20] # Pinky

                    #----- Checking which fingers are up -----
                    fingers = []
                    # Checking the thumb
                    if points[tipIds[0]][0] < points[tipIds[0] - 1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # The rest of the fingers
                    for id in range(1, 5):
                        if points[tipIds[id]][1] < points[tipIds[id] - 2][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    #----- Selection Mode - Two fingers are up -----
                    nonSel = [0, 3, 4] # indexes of the fingers that must be down in the Selection Mode
                    if (fingers[1] and fingers[2]) and all(fingers[i] == 0 for i in nonSel):
                        xp, yp = [x1, y1]

                        # Selecting the colors and the eraser on the screen
                        if(y1 < 125):
                            if(170 < x1 < 295):
                                header = overlayList[0]
                                drawColor = (0, 0, 255)
                            elif(436 < x1 < 561):
                                header = overlayList[1]
                                drawColor = (255, 0, 0)
                            elif(700 < x1 < 825):
                                header = overlayList[2]
                                drawColor = (0, 255, 0)
                            elif(980 < x1 < 1105):
                                header = overlayList[3]
                                drawColor = (0, 0, 0)

                        cv2.rectangle(image, (x1-10, y1-15), (x2+10, y2+23), drawColor, cv2.FILLED)

                    #----- Stand by Mode - Checking when the index and the pinky fingers are open and dont draw
                    nonStand = [0, 2, 3] # indexes of the fingers that need to be down in the Stand Mode
                    if (fingers[1] and fingers[4]) and all(fingers[i] == 0 for i in nonStand):
                        # The line between the index and the pinky indicates the Stand by Mode
                        cv2.line(image, (xp, yp), (x4, y4), drawColor, 5) 
                        xp, yp = [x1, y1]

                    #----- Draw Mode - One finger is up
                    nonDraw = [0, 2, 3, 4]
                    if fingers[1] and all(fingers[i] == 0 for i in nonDraw):
                        # The circle in the index finger indicates the Draw Mode
                        cv2.circle(image, (x1, y1), int(thickness/2), drawColor, cv2.FILLED) 
                        if xp==0 and yp==0:
                            xp, yp = [x1, y1]
                        # Draw a line between the current position and the last position of the index finger
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                        # Update the last position
                        xp, yp = [x1, y1]

                    ## Clear the canvas when the hand is closed
                    if all(fingers[i] == 0 for i in range(0, 5)):
                        imgCanvas = np.zeros((height, width, 3), np.uint8)
                        xp, yp = [x1, y1]


        # Setting the header in the video

        image[0:125, 0:width] = header

        # The image processing to produce the image of the camera with the draw made in imgCanvas
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(image, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        cv2.imshow('Air Canvas', img)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()
