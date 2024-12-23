import cv2 
import numpy as np
from PIL import Image
from ciecam02 import CIECAM02 
import matplotlib.pyplot as plt
from parameters import constants, matrices, colordata

# ===== Example Conversion Functions for Testing ===== #
def xyz2rgb(xyz):
    xyz = xyz/100.0

    M_1 = np.array([[3.2406, -1.5372, -0.4986],
                    [-0.9689, 1.8758,  0.0415],
                    [0.0557, -0.2040,  1.0570]]).T
    RGB = xyz.dot(M_1)
    RGB = np.where(RGB <= 0, 0.00000001, RGB)
    RGB = np.where(
        RGB > 0.0031308,
        1.055*(RGB**0.4166666)-0.055,
        12.92*RGB)

    RGB = np.around(RGB*255)
    RGB = np.where(RGB <= 0, 0, RGB)
    RGB = np.where(RGB > 255, 255, RGB)
    RGB = RGB.astype('uint8')

    return RGB

def rgb2xyz(color):
    color = color/255.0
    color = np.where(
        color > 0.04045,
        np.power(((color+0.055)/1.055), 2.4),
        color/12.92)
    M = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]]).T

    return color.dot(M)*100

# ========== Example Run ========== #
if __name__ == "__main__":

    # forward pass 
    # im = Image.open("./Lenna.png")
    # rgb = np.array(im)
    
    # Load the image
    image_bgr = cv2.imread("./Lenna.png")

    # Convert the image from BGR to RGB if it exists. 
    if image_bgr is not None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  
    else:
        raise FileNotFoundError("Image does not exist.") 
    shape = image_rgb.shape 

    model = CIECAM02(constants, matrices, colordata) # default configuration
    # Configure the model according to the surrounding condition
    model.configure(white="white", surround="average", light="high", bg="high")

    # Forward pass 
    XYZ = rgb2xyz(image_rgb.reshape(-1, 3)) 
    JCh = model.xyz_to_ciecam02(XYZ) 
#Possible error: model returns JQH, here noted JCH

#EDIT: change white point for inverse operation    
    model.configure(white="c", surround="average", light="high", bg="high")    

    # Reverse back to a new image
    RGB = xyz2rgb(model.inverse_model(JCh)).reshape(shape)
    # rgb = jch2rgb(JCH).reshape(shape)



    # Create a figure with two subplots side by side
    plt.figure(figsize=(10, 5))  # Adjust the figure size

    # Show the first image
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axes
    plt.title("original image")  # Optional title

    # Show the second image
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.imshow(RGB)
    plt.axis('off')  # Hide axes
    plt.title("through NO-OP color model")  # Optional title

    # Display the images
    plt.tight_layout()  # Adjust spacing
    plt.show()