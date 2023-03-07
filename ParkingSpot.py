OCCUPIED_PARKING_SPOT_COLOR = (0, 0, 255)
FREE_PARKING_SPOT_COLOR = (0, 255, 0)
HANDICAPPED_PARKING_SPOT_COLOR = (227, 219, 171)
NON_VALID_PARKING_SPOT_COLOR = (0, 156, 255)


class ParkingSpot:
    parking_spot_id = 0
    occupied = False
    coordinates = []
    handicapped = False
    valid = True

    def __init__(self, parking_spot_id, is_occupied, coordinates, is_handicapped, is_valid):
        self.color = None
        self.occupied = is_occupied
        self.coordinates = coordinates
        self.parking_spot_id = parking_spot_id
        self.handicapped = is_handicapped
        self.valid = is_valid

    def is_occupied(self):
        return self.occupied

    def set_occupied(self, is_occupied):
        self.occupied = is_occupied
        self.color = OCCUPIED_PARKING_SPOT_COLOR if is_occupied else FREE_PARKING_SPOT_COLOR
        if self.handicapped and not self.occupied:
            self.color = HANDICAPPED_PARKING_SPOT_COLOR
        if not self.valid and not self.occupied:
            self.color = NON_VALID_PARKING_SPOT_COLOR

    def get_coordinates(self):
        return self.coordinates

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates

    def get_color(self):
        return self.color

    def set_color(self, color):
        self.color = color

    def get_id(self):
        return self.parking_spot_id

    def is_handicapped(self):
        return self.handicapped
