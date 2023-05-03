from ObjectDetection import ObjectDetection

if __name__ == '__main__':
    object_detection = ObjectDetection('window', 'rtsp://tapoadmin:tapopassword123@10.10.120.46:554/stream1')
    object_detection.start()