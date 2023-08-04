# importing packages
import cv2
import mediapipe as mp
from math import hypot
import ctypes
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import time
import htm as ht
import autopy#keyboard and mouse tarcking

# used to convert protobuf message to a dictionary
from google.protobuf.json_format import MessageToDict

# building the model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

mpDraw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volMin, volMax = volume.GetVolumeRange()[:2]

pTime = 0
width = 640
height = 480
frameR = 100
smoothening = 8
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
opened=False
closed=False
# reading from webcam
webcam = cv2.VideoCapture(0)
webcam.set(3, width)
webcam.set(4, height)
prev_vol=0
detector = ht.handDetector(maxHands=1)
screen_width, screen_height = autopy.screen.size()
captured=True
def minimize_maximize_window():
    ctypes.windll.user32.keybd_event(0x5B, 0, 0, 0)  # Left Windows Key Down
    ctypes.windll.user32.keybd_event(0x44, 0, 0, 0)  # 'D' Key Down
    ctypes.windll.user32.keybd_event(0x44, 0, 2, 0)  # 'D' Key Up
    ctypes.windll.user32.keybd_event(0x5B, 0, 2, 0)  # Left Windows Key Up
def click_print_screen():
    ctypes.windll.user32.keybd_event(0x2C, 0, 0, 0)  # Print Screen Key Down
    ctypes.windll.user32.keybd_event(0x2C, 0, 2, 0)  # Print Screen Key Up

while True:
    success, img = webcam.read()

    # flipping the image for model as camera caputers reverse image
    original_img=img.copy() # copying the original image
    img = cv2.flip(img, 1)

    # converting to RGB for model (model needs RGB img)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # passing the image to the media pipe model
    results = hands.process(RGB_img)

    # if there is any result (if any hand is detected)
    if results.multi_hand_landmarks:

        if len(results.multi_handedness) == 2: # if two hands exist in the image
            cv2.putText(original_img, 'Both Hands', (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        else: # if only one hand exists in the image
            for i in results.multi_handedness:
                label = MessageToDict(i)['classification'][0]['label']

                if label == 'Left':
                    captured = True
                    cv2.putText(img, 'Left Hand', (460, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    lmList = []
                    if results.multi_hand_landmarks:
                        for handlandmark in results.multi_hand_landmarks:
                            for id, lm in enumerate(handlandmark.landmark):
                                h, w, c = img.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                lmList.append([id, cx, cy])
                            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

                        x1, y1 = lmList[4][1], lmList[4][2]
                        x2, y2 = lmList[8][1], lmList[8][2]

                        cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
                        cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)

                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                        length = hypot(x2 - x1, y2 - y1)

                        vol = np.interp(length, [15, 220], [volMin, volMax])
                        print(vol, length)
                        if (vol > prev_vol):
                            cv2.putText(img, 'Volume Increasing', (60, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (0, 255, 255), 2)
                        elif(vol < prev_vol):
                            cv2.putText(img, 'Volume Decreasing', (60, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (0, 255, 255), 2)
                        prev_vol = vol
                        volume.SetMasterVolumeLevel(vol, None)

                if label == 'Right':
                    img = original_img.copy()
                    cv2.putText(img, 'Right Hand', (460, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    img = detector.findHands(img)
                    lmlist, bbox = detector.findPosition(img)

                    if len(lmlist) != 0:
                        x1, y1 = lmlist[8][1:]
                        x2, y2 = lmlist[12][1:]

                        fingers = detector.fingersUp()
                        cv2.rectangle(img, (frameR, frameR), (width - frameR, height - frameR), (255, 0, 255), 2)
                        if fingers[1] == 1 and fingers[2] == 0:
                            captured = True
                            cv2.putText(img, 'cursor active', (60, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            x3 = np.interp(x1, (frameR, width - frameR), (0, screen_width))
                            y3 = np.interp(y1, (frameR, height - frameR), (0, screen_height))

                            curr_x = prev_x + (x3 - prev_x) / smoothening
                            curr_y = prev_y + (y3 - prev_y) / smoothening

                            autopy.mouse.move(screen_width - curr_x, curr_y)
                            cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
                            prev_x, prev_y = curr_x, curr_y
                        if all(element == 1 for element in fingers):
                            captured = True
                            if(closed):
                                closed=False
                                minimize_maximize_window()
                            opened=True
                            length, img, lineInfo = detector.findDistance(8, 12, img)
                            if length < 30:
                                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                                autopy.mouse.click()
                        if all(element == 0 for element in fingers) and opened:
                            opened=False
                            closed=True
                            captured = True
                            minimize_maximize_window()
                        if fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1 and all(element == 0 for element in fingers[3:]):
                            length, img, lineInfo = detector.findDistance(8, 12, img)
                            captured = True
                            if length < 30:
                                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                                autopy.mouse.click()
                        if fingers[0]==1 and fingers[4]==1 and fingers[2]==0 and fingers[3]==0 and fingers[1]==0 and captured:
                            detector.highlight(4, 20, img)
                            click_print_screen()
                            cv2.putText(img, 'captured', (60, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            captured=False



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
