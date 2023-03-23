import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

path_to_image = 'inputs/xx.png'
# image = Image.open(path_to_image).convert('RGB')
# image = np.array(image)
image = cv2.cvtColor(cv2.imread(path_to_image), cv2.COLOR_BGR2RGB)
sr_image = model.predict(image)
sr_image = Image.fromarray(sr_image)
sr_image.save('results/xx.png')