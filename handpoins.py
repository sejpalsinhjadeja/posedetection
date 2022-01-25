import cv2
import mediapipe as mp
mpHands = mp.solutions.hands
mpDrow = mp.solutions.drawing_utils
hand = mpHands.Hands()
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hand.process(imgRGB)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDrow.draw_landmarks(img,hand_landmarks,mpHands.HAND_CONNECTIONS)
    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
