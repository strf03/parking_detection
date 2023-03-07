import numpy as np
from shapely.geometry import Polygon


class ParkingDetector:
    THRESHOLD = 0.2
    IOU_THRESHOLD = 35

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def detect_parking(self, cars_coordinates, parking_spots):
        for i, parking_spot in enumerate(parking_spots):
            parking_spot_poly = Polygon(parking_spot.get_coordinates())
            parking_spot.set_occupied(False)
            for cord in cars_coordinates:
                if cord[4] >= self.THRESHOLD:
                    x1, y1, x2, y2 = int(cord[0] * self.width), int(
                        cord[1] * self.height), int(cord[2] * self.width), int(
                        cord[3] * self.height)
                    car_poly = Polygon([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
                    polygon_intersection = parking_spot_poly.intersection(car_poly).area
                    polygon_union = parking_spot_poly.union(car_poly).area
                    iou = round((polygon_intersection / polygon_union) * 100)
                    if iou >= self.IOU_THRESHOLD:
                        parking_spot.set_occupied(True)
