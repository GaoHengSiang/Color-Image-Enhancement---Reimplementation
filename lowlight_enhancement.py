import cv2 
import numpy as np
from PIL import Image
from ciecam02 import CIECAM02 
import matplotlib.pyplot as plt
from parameters import constants, matrices, colordata
from low_backlight import *

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