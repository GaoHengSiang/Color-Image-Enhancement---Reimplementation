import cv2 
import numpy as np
from PIL import Image
from ciecam02 import CIECAM02 
import matplotlib.pyplot as plt
from parameters import constants, matrices, colordata
from low_backlight import *
import argparse

if __name__ == "__main__":
    """
    Usage: python .\lowlight_enhancement.py -i .\test_images\fruits.png
    """
    parser = argparse.ArgumentParser(description="Process an image file.")
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to the input image file."
    )
    
    # Parse arguments
    args = parser.parse_args()
    input_file = args.input
    
    # Load the image
    image_bgr = cv2.imread(input_file)

    # Convert the image from BGR to RGB if it exists. 
    if image_bgr is not None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  
    else:
        raise FileNotFoundError("Image does not exist.") 
    shape = image_rgb.shape 

    model = CIECAM02(constants, matrices, colordata) # default configuration

    # Forward pass 
    print("Convert RGB_i to XYZ_i...")
    RGB_i = image_rgb/255.0
    XYZ_i = device_rgb_to_xyz(RGB_i, M_f, gamma_rf, gamma_gf, gamma_bf) 

    # Configure the model according to the surrounding condition
    print("Forward pass through the color model...")
    model.configure(white="full_light", surround="average", light="high", bg="high")
    XYZ_i = XYZ_i.reshape(-1, 3)
    JCh = model.xyz_to_ciecam02(XYZ_i) 

    # Reverse back to a new image
    print("Reverse pass through the color model...")
    model.configure(white="low_light", surround="average", light="high", bg="high")  
    XYZ_e = model.inverse_model(JCh).reshape(shape)

    #post gamut mapping
    print("post gamut mapping...")
    JCh_reshape = JCh.reshape(shape)
    JC = JCh_reshape[..., 0] * JCh_reshape[..., 1]
    # Normalize array to [0, 1] range
    normalized_JC = (JC - np.min(JC)) / (np.max(JC) - np.min(JC))
    RGB_prime = post_gamut_mapping(RGB_i, XYZ_e, normalized_JC)
    # rgb = jch2rgb(JCH).reshape(shape)
    print("Simulate low backlight for original image...")
    without_enhancement = simulated_low_backlight(RGB_i, M_l, gamma_rl, gamma_gl, gamma_bl)
    print("Simulate low backlight for enhanced image...")
    with_enhancement = simulated_low_backlight(RGB_prime, M_l, gamma_rl, gamma_gl, gamma_bl)

    #print("Other test")
    #RGB_test = device_xyz_to_rgb(XYZ_e, M_l, gamma_rl, gamma_gl, gamma_bl)
    #RGB_test = simulated_low_backlight(RGB_test, M_l, gamma_rl, gamma_gl, gamma_bl)
    
    # Create a figure with two subplots side by side
    plt.figure(figsize=(10, 5))  # Adjust the figure size

    # Show the first image
    plt.subplot(1, 3, 1)  # 1 row, 2 columns, first subplot
    plt.imshow(RGB_i)
    plt.axis('off')  # Hide axes
    plt.title("original image")  # Optional title

    plt.subplot(1, 3, 2)  # 1 row, 2 columns, first subplot
    plt.imshow(without_enhancement)
    plt.axis('off')  # Hide axes
    plt.title("without enhancement")  # Optional title

    # Show the second image
    plt.subplot(1, 3, 3)  # 1 row, 2 columns, second subplot
    plt.imshow(with_enhancement)
    plt.axis('off')  # Hide axes
    plt.title("with enhancement")  # Optional title

    # Display the images
    plt.tight_layout()  # Adjust spacing
    plt.show()