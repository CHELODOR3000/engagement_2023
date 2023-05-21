import cv2
import mediapipe as mp
import numpy as np
import time
import math


RIGHT_EYE_INDEXES = [33, 133, 145, 159] # порядок важен
LEFT_EYE_INDEXES = [263, 362, 374, 386] # порядок важен
FACE_INDEXES = [33, 263, 1, 61, 291, 199]

def get_blink_ratio(img, right_eye_coords, left_eye_coords):
    # right eye horizontal line 
    rh_right = right_eye_coords[0] # 33
    rh_left = right_eye_coords[1] # 133

    # right eye vertical line 
    rv_top = right_eye_coords[3] # 159
    rv_bottom = right_eye_coords[2] # 145

    cv2.line(img, rh_right, rh_left, (0, 0, 255), 2)
    cv2.line(img, rv_top, rv_bottom, (0, 0, 255), 2)

    # left eye horizontal line 
    lh_right = left_eye_coords[1] # 362
    lh_left = left_eye_coords[0] # 263

    # left eye vertical line 
    lv_top = left_eye_coords[3] # 386
    lv_bottom = left_eye_coords[2] # 374

    rhDistance = get_euclaidean_distance(rh_right, rh_left)
    rvDistance = get_euclaidean_distance(rv_top, rv_bottom)

    lvDistance = get_euclaidean_distance(lv_top, lv_bottom)
    lhDistance = get_euclaidean_distance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio 

def get_euclaidean_distance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# print(os.listdir(r'../'))
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r'test_my_face.MOV')
# output = cv2.VideoWriter('my_face_processed.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640, 480))

while cap.isOpened():
    _, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    print(image.shape)
    face_3d = []
    face_2d = []

    left_eye_2d = []
    right_eye_2d = []

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:
            # у каждого кадра берем только 6 нужных нам точек
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in FACE_INDEXES:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

                if idx in RIGHT_EYE_INDEXES:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    right_eye_2d.append([x, y])

                if idx in LEFT_EYE_INDEXES:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    left_eye_2d.append([x, y])

        # blink ratio
        ratio = get_blink_ratio(image, right_eye_2d, left_eye_2d) 
        print(ratio)   

        # Convert it to the NumPy array
        face_2d = np.array(face_2d, dtype=np.float64)
        
        # Convert it to the NumPy array
        face_3d = np.array(face_3d, dtype=np.float64)

        # The camera matrix
        focal_length = 2.2

        # Have to calibrate camera to get cam_matrix
        cam_matrix = np.array([ [focal_length, 0, img_h / 2], 
                                [0, focal_length, img_w / 2], 
                                [0, 0, 1]])

        # The distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        _, rot_vec, translation_vector = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Get rotation matrix
        rmat, _ = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360
        
        # See where the user's head tilting
        if ratio > 3.6:
            text = 'Not engaged'
        else:
            if y > 7 or y < -15 or x < -3:
                text = 'Not engaged'
            else:
                text = 'Engaged'
            

        # Display the nose direction
        # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

        #p1 = (int(nose_2d[0]), int(nose_2d[1]))
        #p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
        
        #cv2.line(image, p1, p2, (255, 0, 0), 3)

        # Add the text on the image
        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        cv2.putText(image, "x: " + str(np.round(x, 1)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "y: " + str(np.round(y, 1)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(image, "z: " + str(np.round(z,1)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        end = time.time()
        totalTime = end - start

        # fps = 1 / totalTime
        
        #print("FPS: ", fps)

        # cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        '''mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)'''

    # output.write(image)
    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break


cap.release()