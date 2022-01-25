import cv2
import mediapipe as mp
mpPose = mp.solutions.pose
mpDrow = mp.solutions.drawing_utils
pose = mpPose.Pose()
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDrow.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
