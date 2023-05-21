# Program constructs Concentration Index and returns a classification of engagement.
# coding: utf-8
import cv2
import numpy as np
import os
import sys
sys.path.append('util')
import torch
import torch.nn as nn
import torchvision
import mtcnn


class Analysis:
    def __init__(self):
        self.device = 'cpu' # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.face_detector = mtcnn.MTCNN() # FacialImageProcessing(False)
        self.gaze_tracker = self.load_gaze_tracker('util/model/enet_b0_sota.pt')
        self.sliding_window_size = 2
        self.sliding_window_preds = []
        self.decode_target = {
            0: 'upper left part of the windshield', # 'левая верхняя часть лобового стекла',
            1: 'straight', # 'прямо перед собой',
            2: 'speedometer', #'спидометр',
            3: 'radio', # 'радио',
            4: 'upper right part of the windshield', #'правая верхняя часть лобового стекла',
            5: 'bottom right part of the windshield', #'правая нижняя часть лобового стекла', 
            6: 'right side mirror', #'правое боковое зеркало',
            7: 'rear view mirror', #'зеркало заднего вида',
            8: 'left side mirror', #'левое боковое зеркало',
        }

        self.frame_count = 0
        
    def load_gaze_tracker(self, model_path: str):
        model = torch.load(model_path, map_location = torch.device('cpu'))
        # model = torchvision.models.efficientnet_b0(pretrained = False)
        # model.classifier = nn.Linear(1280, 9, bias = True)
        # model = torchvision.models.alexnet(pretrained = False)
        # model.classifier[6] = nn.Linear(4096, 9)
        if self.device == 'cuda:0':
            model.to(self.device)
        # model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
        model.eval()

        return model

    def track_gaze(self, frame):
        # frame = frame.copy()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print(type(frame_bgr))
        bounding_boxes = self.face_detector.detect_faces(frame_bgr)
        
        if len(bounding_boxes) == 0:
            return frame
        
        x, y, width, height = bounding_boxes[0]['box']
        print(x, y, width, height)
        # сохранение лица
        face = frame[y:y+height,x:x+width,:]

        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), 1)

        '''transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(p=1)])'''

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((260, 260)),
                # torchvision.transforms.RandomHorizontalFlip(p=1), # оставить при работе с вебки
                # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            # std=[0.229, 0.224, 0.225])
            ]
        )

        face_transformed = transform(face)
        face_transformed = torch.unsqueeze(face_transformed, 0)

        if self.device == 'cuda:0':
            face_transformed.to(self.device)

        # print(face_transformed.shape)
        # print(torch.unsqueeze(face_transformed, 0).shape)
        predicted_gaze_direction = self.gaze_tracker(face_transformed)

        probs = nn.functional.softmax(predicted_gaze_direction, dim=1)# .topk(9, dim = 1)
        print(probs[0].detach())

        # print(np.around(probs[0].detach().numpy(), decimals = 2))

        predicted_gaze_direction = predicted_gaze_direction.argmax(1).cpu().detach().item()
        print(predicted_gaze_direction)
        # print(predicted_gaze_direction, type(predicted_gaze_direction))
        if len(self.sliding_window_preds) < self.sliding_window_size:
            self.sliding_window_preds.append(predicted_gaze_direction)
            predicted_gaze_direction = 'Loading...'
            print(len(self.sliding_window_preds))
        else:
            self.sliding_window_preds = self.sliding_window_preds[1:] + [predicted_gaze_direction]
            print(len(self.sliding_window_preds))
            predicted_gaze_direction = np.round(np.bincount(self.sliding_window_preds).argmax())
            print(predicted_gaze_direction)
            predicted_gaze_direction = self.decode_target[predicted_gaze_direction]


        # predicted_gaze_direction = self.decode_target[predicted_gaze_direction]


        cv2.putText(frame, predicted_gaze_direction, (50, 100), self.font, 1, (0, 0, 255), 3)
        
        return frame

