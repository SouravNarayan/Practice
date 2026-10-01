import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Get the path of the folder where ComputerVision.py is saved
script_dir = Path(__file__).parent

# Target the image inside your new images folder
filename = str(script_dir / "images" / "Xray.jpg")

# Load the image
img = cv2.imread(filename)


# filename=
# img= cv2.imread(filename)
[m,n,c] = img.shape
img1= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
b= img[:, :, 0]
g= img[:, :, 1]
r= img[:, :, 2]

grey_img=np.uint8(np.zeros([m,n]))

for i in range(m):
    for j in range(n):
        grey_value=0.114*b[i,j] + 0.587*g[i,j] + 0.299*r[i,j]
        grey_img[i,j]=grey_value

Neg_img=np.uint8(np.zeros([m,n]))

for i in range(m):
    for j in range(n):
        Neg=255-grey_img[i,j]
        Neg_img[i,j]=Neg

plt.figure(figsize=(15,5))

plt.subplot(131)
plt.imshow(img1)
plt.title('Original image')
plt.axis('off')

plt.subplot(132)
plt.imshow(grey_img, cmap='gray')
plt.title('Grey_Scale image')
plt.axis('off')

plt.subplot(133)
plt.imshow(Neg_img, cmap='gray')
plt.title('Negative image')
plt.axis('off')

plt.tight_layout()
plt.savefig('Output.png')

