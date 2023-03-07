from ParkingPlacesCreator import ParkingPlacesCreator
from ObjectDetection import ObjectDetection

from fastapi import Request, FastAPI
import uvicorn
from threading import Thread

app = FastAPI()
object_detection = ObjectDetection('window', 'rtsp://tapoadmin:tapopassword123@10.10.120.46:554/stream1')


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/api/lot/general_info")
async def general_info():
    all_spots = object_detection.parking_lot.get_all_parking_space_count()
    occupied_spots = object_detection.parking_lot.get_occupied_count()
    free_spots = all_spots - occupied_spots
    return {
        "All parking spots": all_spots,
        "Free parking spots": free_spots,
        "Occupied parking spots": occupied_spots,
    }


@app.get("/api/lot/{spot_id}")
async def get_spot(spot_id: int):
    parking_spot = object_detection.parking_lot.get_parking_spot(spot_id)
    if parking_spot is None:
        return {"Parking spot with id " + str(spot_id) + " does not exist"}
    return {
        "Id": parking_spot.get_id(),
        "Occupied": parking_spot.is_occupied(),
        "Color": parking_spot.get_color(),
        "Handicapped": parking_spot.is_handicapped()
    }


@app.post("/api/lot/add")
async def add_spot(request: Request):
    json_request = await request.json()
    parking_spot = object_detection.parking_lot.get_parking_spot(json_request['spot_id'])
    if parking_spot is not None: # todo what if coordinates or other information missing
        return {"Parking spot with id " + str(json_request['spot_id']) + " already exists"}
    object_detection.parking_lot.add_parking_spot(json_request['spot_id'],
                                                  json_request['coordinates'],
                                                  json_request['handicapped'],
                                                  json_request['valid'])
    return {
        "Result": "successfully added",
    }


@app.post("/api/lot/delete/{spot_id}")
async def delete_spot(spot_id: int):
    parking_spot = object_detection.parking_lot.get_parking_spot(spot_id)
    if parking_spot is None:
        return {"Parking spot with id " + str(spot_id) + " does not exist"}
    object_detection.parking_lot.delete_parking_spot(spot_id)
    return {
        "Result": "Successfully deleted",
    }


def run_server():
    uvicorn.run(app, host='localhost', port=8080)


def run_detection():
    object_detection.start()


if __name__ == '__main__':
    t1 = Thread(target=run_server)
    t2 = Thread(target=run_detection)

    t1.start()
    t2.start()
    # ParkingPlacesCreator('rtsp://tapoadmin:tapopassword123@10.10.120.46:554/stream1').start()
