"""
Запуск прототипа для классификации направления взгляда водителя через веб-камеру
"""
from util.analysis_realtime import Analysis
import cv2

cap = cv2.VideoCapture(0)
ana = Analysis()

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    processed_frame = ana.track_gaze(frame)

    cv2.imshow("Frame", processed_frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
