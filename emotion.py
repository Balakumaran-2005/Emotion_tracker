import cv2
from facial_emotion_recognition import EmotionRecognition
er=EmotionRecognition(device='cpu')
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame=er.recognise_emotion(frame,return_type='BGR')
    cv2.imshow("Emotion_tracker",frame)
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
