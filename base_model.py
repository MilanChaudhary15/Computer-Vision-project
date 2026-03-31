import os
import torch
from collections import OrderedDict

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def save_network(self, model, name):
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.save_dir, f"{name}.pth"))

    def load_network(self, model, name):
        path = os.path.join(self.save_dir, f"{name}.pth")
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device))

    def get_current_losses(self, loss):
        return OrderedDict({"loss": float(loss)})
