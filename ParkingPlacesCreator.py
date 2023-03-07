import cv2
import numpy as np
import json

from PlottingManager import PlottingManager
from YoloModel import YoloModel
import time

FINAL_LINE_COLOR = (0, 255, 0)
WORKING_LINE_COLOR = (0, 0, 255)

ASCII_ESC = 27
ASCII_S = 115
ASCII_I = 105
ASCII_D = 100

WIDTH = 440
HEIGHT = 100

INITIAL_PARKING_SPOT_ARRAY = [{'id': 1, 'coordinates': [], 'handicapped': False, 'valid': True}]


class ParkingPlacesCreator:
    def __init__(self, link):
        self.window_name = 'window'
        self.done = False
        self.current = [0, 0]
        self.parking_spots = INITIAL_PARKING_SPOT_ARRAY  # todo add is_occupied
        self.link = link
        self.model = YoloModel('ultralytics/yolov5', 'yolov5m')
        self.plotting_manager = PlottingManager(1498, 720)  # todo constant
        self.cords = []

    def plot_parking_spots(self, frame):
        if len(self.parking_spots) > 0:
            for parking_spot in self.parking_spots:
                if len(parking_spot["coordinates"]) == 4:
                    cv2.polylines(frame, [np.array(parking_spot["coordinates"])], True,
                                  FINAL_LINE_COLOR, 1)
                else:
                    cv2.polylines(frame, [np.array(parking_spot["coordinates"])], False,
                                  WORKING_LINE_COLOR, 1)
                if self.parking_spots[-1]['coordinates'] and len(self.parking_spots[-1]['coordinates']) != 4:
                    cv2.line(frame, tuple(self.parking_spots[-1]['coordinates'][-1]), self.current,
                             WORKING_LINE_COLOR)
        return frame

    def delete_last_rect(self):
        self.parking_spots.pop(-1)
        if len(self.parking_spots) == 0:
            self.parking_spots = INITIAL_PARKING_SPOT_ARRAY

    def on_mouse(self, event, x, y, buttons, user_param):
        if self.done:
            return

        if event == cv2.EVENT_MOUSEMOVE:
            self.current = [x, y]
        elif event == cv2.EVENT_LBUTTONDOWN:
            parking_spot_id = self.parking_spots[-1]['id'] + 1
            new_parking_spot = {'id': parking_spot_id, 'coordinates': [[x, y]], 'handicapped': False, 'valid': True}
            if len(self.parking_spots[-1]['coordinates']) == 4:
                self.parking_spots.append(new_parking_spot)
            else:
                self.parking_spots[-1]['coordinates'].append([x, y])

            print(self.parking_spots)

    def start(self):
        if self.link:
            cap = cv2.VideoCapture(self.link)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 2)
        else:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        last_recorded_time_score_frame = time.time()
        while True:
            curr_time = time.time()
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1498, 720))
            frame[HEIGHT:WIDTH, :] = 0  # black stripe
            if curr_time - last_recorded_time_score_frame >= 2.0:  # score frame every 2 seconds
                self.cords = self.model.score_frame(frame)
                last_recorded_time_score_frame = curr_time
            frame = self.plotting_manager.plot_car_boxes(self.cords, frame)
            frame = self.plot_parking_spots(frame)
            cv2.imshow(self.window_name, frame)

            c = cv2.waitKey(1)
            if c == ASCII_ESC:
                break
            elif c == ASCII_D:
                self.delete_last_rect()
            elif c == ASCII_I:
                self.interrupt_drawing()
            elif c == ASCII_S:
                self.save_to_json()

        cap.release()
        cv2.destroyAllWindows()

    def save_to_json(self):
        print('Saving to JSON...')
        with open('final_kunratice_real.json', 'w') as f:
            json.dump(self.parking_spots, f, indent=4)

    def interrupt_drawing(self):
        if len(self.parking_spots[-1]['coordinates']) != 4:
            self.parking_spots.pop(-1)
        if len(self.parking_spots) == 0:
            self.parking_spots = INITIAL_PARKING_SPOT_ARRAY
