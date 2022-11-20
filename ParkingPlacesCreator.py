import cv2
import numpy as np
import json

FINAL_LINE_COLOR = (0, 255, 0)
WORKING_LINE_COLOR = (0, 0, 255)


class ParkingPlacesCreator:
    def __init__(self):
        self.window_name = 'window'
        self.done = False
        self.current = [0, 0]
        self.parking_spots = [{'id': 1, 'coordinates': [], 'handicapped': False}] # todo add is_occupied

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
            self.parking_spots = [{'id': 1, 'coordinates': [], 'handicapped': False}]

    def on_mouse(self, event, x, y, buttons, user_param):
        if self.done:
            return

        if event == cv2.EVENT_MOUSEMOVE:
            self.current = [x, y]
        elif event == cv2.EVENT_LBUTTONDOWN:
            parking_spot_id = self.parking_spots[-1]['id'] + 1
            new_parking_spot = {'id': parking_spot_id, 'coordinates': [[x, y]], 'handicapped': False}
            if len(self.parking_spots[-1]['coordinates']) == 4:
                self.parking_spots.append(new_parking_spot)
            else:
                self.parking_spots[-1]['coordinates'].append([x, y])

            print(self.parking_spots)

    def start(self, link):
        if link:
            cap = cv2.VideoCapture(link)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 2)
        else:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (960, 540))
            frame = self.plot_parking_spots(frame)
            cv2.imshow(self.window_name, frame)

            c = cv2.waitKey(1)
            if c == 27:
                break
            if c == 0:  # todo ascii DEC column - use constants
                self.delete_last_rect()
            if c == 105:
                self.interrupt_drawing()
            if c == 115:
                self.save_to_json()

        cap.release()
        cv2.destroyAllWindows()

    def save_to_json(self):
        print('Saving to JSON...')
        with open('parking_lot.json', 'w') as f:
            json.dump(self.parking_spots, f, indent=4)

    def interrupt_drawing(self):
        if len(self.parking_spots[-1]['coordinates']) != 4:
            self.parking_spots.pop(-1)
