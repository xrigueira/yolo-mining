import cv2
import torch
print(cv2.__version__)
print(cv2.cuda.getCudaEnabledDeviceCount())

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(device)