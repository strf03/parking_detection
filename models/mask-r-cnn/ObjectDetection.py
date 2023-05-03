import os
import cv2
import numpy as np
import time

class ObjectDetection:
    WIDTH = 1920
    HEIGHT = 1080
    MINUTE_IN_SECONDS = 60

    confThreshold = 0.4
    maskThreshold = 0.3

    def __init__(self, window_name, source):
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        self.finish = False
        self.window_name = window_name
        self.source = source
        self.cords = []
        self.classes = None
        self.colors = []

    def get_free_count(self):
        return self.parking_lot.get_all_parking_space_count() - self.parking_lot.get_occupied_count()

    def start(self):
        if self.source:
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        else:
            cap = cv2.VideoCapture(0)

        cv2.namedWindow(self.window_name)

        last_recorded_time_score_frame = time.time()

        classesFile = "mask_rcnn/object_detection_classes_coco.txt"

        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        print(self.classes)
        # Load the colors
        colorsFile = "mask_rcnn/colors.txt"
        with open(colorsFile, 'rt') as f:
            colorsStr = f.read().rstrip('\n').split('\n')

        for i in range(len(colorsStr)):
            rgb = colorsStr[i].split(',')
            color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
            self.colors.append(color)

        # Give the textGraph and weight files for the model
        textGraph = "mask_rcnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        modelWeights = "mask_rcnn/frozen_inference_graph.pb"

        # Load the network
        net = cv2.dnn.readNetFromTensorflow(modelWeights, textGraph)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        while True:
            curr_time = time.time()
            ret, frame = cap.read()

            frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
            frame[100:650, :] = 0  # black stripe
            """frame[100:440, :] = 0  # black stripe

            if curr_time - last_recorded_time_score_frame >= 2.0:  # score frame every 2 seconds
                self.cords = self.model.score_frame(frame)
                last_recorded_time_score_frame = curr_time

            self.plot_information(frame)"""

            # Create a 4D blob from a frame.
            blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)

            # Set the input to the network


            if curr_time - last_recorded_time_score_frame >= 2.0:  # score frame every 2 seconds
                # Run the forward pass to get output from the output layers
                net.setInput(blob)
                boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
                self.postprocess(boxes, masks, frame)

            cv2.imshow(self.window_name, frame)
            c = cv2.waitKey(1)
            if c == 27:
                self.finish = True
                break

        cap.release()
        cv2.destroyAllWindows()

    def postprocess(self, boxes, masks, frame):
        # Output size of masks is NxCxHxW where
        # N - number of detected boxes
        # C - number of classes (excluding background)
        # HxW - segmentation shape
        numClasses = masks.shape[1]
        numDetections = boxes.shape[2]

        frameH = frame.shape[0]
        frameW = frame.shape[1]

        for i in range(numDetections):
            box = boxes[0, 0, i]
            mask = masks[i]
            score = box[2]
            if score > self.confThreshold:
                classId = int(box[1])
                print(classId)

                # Extract the bounding box
                left = int(frameW * box[3])
                top = int(frameH * box[4])
                right = int(frameW * box[5])
                bottom = int(frameH * box[6])

                left = max(0, min(left, frameW - 1))
                top = max(0, min(top, frameH - 1))
                right = max(0, min(right, frameW - 1))
                bottom = max(0, min(bottom, frameH - 1))

                # Extract the mask for the object
                classMask = mask[classId]

                # Draw bounding box, colorize and show the mask on the image
                self.drawBox(frame, classId, score, left, top, right, bottom, classMask)


    def drawBox(self, frame, classId, conf, left, top, right, bottom, classMask):
        # Draw a bounding box.
        #cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

        # Print a label of class.
        label = '%.2f' % conf
        if self.classes:
            assert (classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        """cv2.rectangle(frame, (left, top - round(1.5 * labelSize[1])),
                      (left + round(1.5 * labelSize[0]), top + baseLine),
                      (255, 255, 255), cv2.FILLED)"""
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

        # Resize the mask, threshold, color and apply it on the image
        classMask = cv2.resize(classMask, (right - left + 1, bottom - top + 1))
        mask = (classMask > self.maskThreshold)
        roi = frame[top:bottom + 1, left:right + 1][mask]

        color = self.colors[classId % len(self.colors)]
        # Comment the above line and uncomment the two lines below to generate different instance colors
        # colorIndex = random.randint(0, len(colors)-1)
        # color = colors[colorIndex]

        frame[top:bottom + 1, left:right + 1][mask] = (
                [0.3 * color[0], 0.3 * color[1], 0.3 * color[2]] + 0.7 * roi).astype(np.uint8)

        # Draw the contours on the image
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame[top:bottom + 1, left:right + 1], contours, -1, color, 3, cv2.LINE_8, hierarchy, 100)
