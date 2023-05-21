"""
Запуск прототипа для классификации направления взгляда водителя на изображении
"""
from util.analysis_realtime import Analysis
import cv2


ana = Analysis()

frame = cv2.imread('/home/sergey/Загрузки/Sub156_vid1_frame620.png')
processed_frame = ana.track_gaze(frame)

cv2.imshow('Frame',processed_frame)
cv2.waitKey(0)
