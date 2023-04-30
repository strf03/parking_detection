import torch


class YoloModel:
    model = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, git_hub='WongKinYiu/yolov7', name='yolov7x.pt'):
        """
        Loads Yolo7 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        torch.cuda.current_device()
        torch.cuda._initialized = True
        model = torch.hub.load(git_hub, 'custom', name, verbose=False)
        self.model = model
        self.model.classes = [2, 3, 5, 7]  # 2 car, 3 motorcycle, 5 bus, 7 truck

    def score_frame(self, frame):
        """
        Scores frame.
        :return: return coordinates of objects in frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        cords = results.xyxyn[0][:, :-1].cpu().numpy()
        return cords
