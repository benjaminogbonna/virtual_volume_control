import time
import math
import numpy as np
import mediapipe as mp
import cv2
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import hand_tracking_module as htm


cam_w, cam_h = 640, 480
cap = cv2.VideoCapture(0)

cap.set(3, cam_w)
cap.set(4, cam_h)

detector = htm.HandDetector(detection_con=.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol_range = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20., None)
min_vol = vol_range[0]
max_vol = vol_range[1]

vol = 0
vol_bar = 400
vol_per = 0
prev_time = 0
while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    land_mark_List = detector.find_position(img, draw=False)

    if len(land_mark_List) > 0:
        # print(land_mark_List[4], land_mark_List[8])
        x1, y1 = land_mark_List[4][1], land_mark_List[4][2]
        x2, y2 = land_mark_List[8][1], land_mark_List[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        print(length)
        vol = np.interp(length, [50, 180], [min_vol, max_vol])
        vol_bar = np.interp(length, [50, 180], [400, 150])
        vol_per = np.interp(length, [50, 180], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)
        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_per)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time

    cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow('Image', img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
