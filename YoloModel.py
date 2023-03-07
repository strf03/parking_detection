import torch


class YoloModel:
    model = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classes = None

    def __init__(self, git_hub, name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        torch.cuda.current_device()
        torch.cuda._initialized = True
        model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7x.pt', verbose=False) # , pretrained=True
        self.model = model
        self.model.classes = [2, 3, 5, 7]  # 0 person, 2 car, 3 motorcycle, 5 bus, 7 truck
        self.classes = self.model.names  # todo is this necessary??

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        cords = results.xyxyn[0][:, :-1].cpu().numpy()
        return cords
