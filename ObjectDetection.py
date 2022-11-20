import os
import cv2
import time

from ParkingDetector import ParkingDetector
from ParkingLot import ParkingLot
from PlottingManager import PlottingManager
from YoloModel import YoloModel


class ObjectDetection:

    def __init__(self, window_name, source):
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        self.finish = False
        self.window_name = window_name
        self.model = YoloModel('ultralytics/yolov5', 'yolov5m')
        self.plotting_manager = PlottingManager(1498, 720)  # todo constant
        self.parking_lot = ParkingLot()
        self.parking_spots = self.parking_lot.get_parking_spots()
        self.parking_detector = ParkingDetector(1498, 720)  # todo constant
        self.source = source
        self.cords = []

        self.x_shape = 1498
        self.y_shape = 720

    def on_mouse(self, event, x, y, buttons, user_param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)

    def get_free_count(self):
        return self.parking_lot.get_all_parking_space_count() - self.parking_lot.get_occupied_count()

    def start(self):
        if self.source:
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        else:
            cap = cv2.VideoCapture(0)

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        last_recorded_time_score_frame = time.time()
        last_recorded_time_parking_spots = time.time()

        while True:
            frame = cv2.imread('savedImageWhole.jpg')
            curr_time = time.time()
            # ret, frame = cap.read()

            """pts1 = np.float32([[0, 430], [1498, 430],
                               [0, 720], [1498, 720]])
            pts2 = np.float32([[0, 0], [1498, 0],
                               [0, 720], [1498, 720]])"""

            frame[100:440, :] = 0  # todo nemazat cerny pruh

            # matrix = cv2.getPerspectiveTransform(pts1, pts2)
            # frame = cv2.warpPerspective(frame, matrix, (1498, 720))

            if curr_time - last_recorded_time_score_frame >= 2.0:  # score frame every 2 seconds
                self.cords = self.model.score_frame(frame)
                last_recorded_time_score_frame = curr_time

            if curr_time - last_recorded_time_parking_spots >= 3.0:  # find IOU every 3 seconds
                self.parking_detector.detectParking(self.cords, self.parking_spots)
                last_recorded_time_parking_spots = curr_time

            frame = self.plotting_manager.plot_boxes(self.cords, frame)
            frame = self.plotting_manager.plot_parking_spots(self.parking_spots, frame)
            # frame = self.plotting_manager.plot_statistics(self.parking_spots, frame)
            cv2.imshow(self.window_name, frame)

            cv2.imshow(self.window_name, frame)
            c = cv2.waitKey(1)
            if c == 27:
                self.finish = True
                break

        cap.release()
        cv2.destroyAllWindows()
