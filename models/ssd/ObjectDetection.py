import time
from torchvision import transforms
import torchvision
import torch
import cv2
import os


from ParkingDetector import ParkingDetector
from ParkingLot import ParkingLot
from PlottingManager import PlottingManager



class ObjectDetection:
    WIDTH = 1920
    HEIGHT = 1080
    MINUTE_IN_SECONDS = 60

    confThreshold = 0.5
    maskThreshold = 0.3
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self, window_name, source):
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        self.finish = False
        self.window_name = window_name
        self.source = source
        self.classes = None
        self.model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.model.eval()
        self.model.to(self.device)
        self.boxes = []
        self.scores = []
        self.labels = []
        self.class_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def get_prediction(self, frame, threshold):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        frame = transform(frame).to(self.device)
        frame = frame.unsqueeze(0)  # add a batch dimension
        pred = self.model(frame.to(self.device))
        pred_class = [self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        return pred_boxes, pred_class

    def object_detection_api(self, frame, threshold=0.5, rect_th=3, text_size=3, text_th=3):
        boxes, pred_cls = self.get_prediction(frame, threshold)
        for i in range(len(boxes)):
            cv2.rectangle(frame, int(boxes[i][0]), int(boxes[i][1]), color=(0, 255, 0), thickness=rect_th)
            cv2.putText(frame, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                        thickness=text_th)
        return frame

    def start(self):
        if self.source:
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        else:
            cap = cv2.VideoCapture(0)

        cv2.namedWindow(self.window_name)
        last_recorded_time_score_frame = time.time()

        while True:
            curr_time = time.time()
            ret, frame = cap.read()

            frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
            frame[100:650, :] = 0  # black stripe

            # Convert the frame to a PyTorch tensor
            img_tensor = self.transform(frame).to(self.device)

            # Pass the tensor through the model to get the predictions
            if curr_time - last_recorded_time_score_frame >= 2.0:  # score frame every 2 seconds
                with torch.no_grad():
                    pred = self.model([img_tensor])
                    self.boxes = pred[0]['boxes'].cpu().numpy().astype(int)
                    self.scores = pred[0]['scores'].cpu().numpy()
                    self.labels = pred[0]['labels'].cpu().numpy()

                last_recorded_time_score_frame = curr_time

            # Draw the predicted boxes on the frame
            self.print_boxes(self.boxes, self.scores, self.labels, frame)
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def print_boxes(self, boxes, scores, labels, frame):
        for i in range(len(boxes)):
            if scores[i] > 0.2:
                x1, y1, x2, y2 = boxes[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                class_name = self.class_names[labels[i]]
                cv2.putText(frame, class_name, (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


