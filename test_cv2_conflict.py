import cv2
import torch

print(cv2.__version__, flush=True)
print(torch.__version__, flush=True)
q, r = torch.linalg.qr(torch.randn(510, 25))
print("done")

