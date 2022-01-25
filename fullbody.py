import cv2
import mediapipe as mp
mpPose = mp.solutions.pose
mpHands = mp.solutions.hands
mpDrow = mp.solutions.drawing_utils
pose = mpPose.Pose()
hand = mpHands.Hands()
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    poseResults = pose.process(imgRGB)
    handResults = hand.process(imgRGB)
    if poseResults.pose_landmarks:
        mpDrow.draw_landmarks(img,poseResults.pose_landmarks,mpPose.POSE_CONNECTIONS)
    if handResults.multi_hand_landmarks:
        for hand_landmarks in handResults.multi_hand_landmarks:
            mpDrow.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)
    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
