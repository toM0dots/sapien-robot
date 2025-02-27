import torch
from PIL import Image

t = torch.load('my_tensor.pt')
print(type(t))
print(t.shape)
print(t.min(), t.max())
image = Image.fromarray(t.squeeze().numpy())

image.save('my_tensor.png')
