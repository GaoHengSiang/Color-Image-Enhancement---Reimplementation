import cv2 
import numpy as np
from PIL import Image
from ciecam02 import CIECAM02 
import matplotlib.pyplot as plt
from parameters import constants, matrices, colordata
from low_backlight import *
from nightlight_sim import *
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
    parser.add_argument(
    "-s", "--strength", 
    type=float, 
    required=True, 
    help="Nightlight strength (0 ~ 100)"
    )
    parser.add_argument(
    "-m", "--map", 
    action="store_true", 
    help="Enable post-gamut mapping (use --no-map to disable)."
    )
    parser.add_argument(
        "--no-map", 
        dest="map", 
        action="store_false", 
        help="Disable post-gamut mapping (use --map to enable)."
    )

    # Parse arguments
    args = parser.parse_args()
    input_file = args.input
    strength = args.strength
    do_map = args.map

    # Load the image
    image_bgr = cv2.imread(input_file)

    # Convert the image from BGR to RGB if it exists. 
    if image_bgr is not None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  
    else:
        raise FileNotFoundError("Image does not exist.") 
    shape = image_rgb.shape 

    model = CIECAM02(constants, matrices, colordata) # default configuration
    
    #dynamically set white point
    temperature = map_windows_nightlight_slider_to_temp(strength)
    channel_gain = kelvin_to_scaling_factors(temperature)
    channel_gain = np.array(channel_gain).reshape(1, 1, 3)
    nl_whitepoint = device_rgb_to_xyz(channel_gain, M_f, gamma_rf, gamma_gf, gamma_bf)
    nl_whitepoint = list(nl_whitepoint.reshape(3))
    constants["whitepoint"]["night_light"] = nl_whitepoint
    fl_white_point = constants["whitepoint"]["full_light"]
    print(f"full light white point: {fl_white_point}")
    print(f"night light white point: {nl_whitepoint}")

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
    model.configure(white="night_light", surround="average", light="high", bg="high")  
    XYZ_e = model.inverse_model(JCh).reshape(shape)

    #Inverse Device Full Light (enhanced xyz back to rgb)
    print("enhanced XYZ to RGB...")
    RGB_nl = device_xyz_to_rgb(XYZ_e, M_f, gamma_rf, gamma_gf, gamma_bf)

    #Inverse Channel Gain
    print("Inverse Channel Gain, handling zero denominator, clipping...")
    safe_gain = np.clip(channel_gain, 1e-5, 1)
    RGB_nl = RGB_nl / safe_gain
    RGB_clip = np.clip(RGB_nl, 0, 1)
    #RGB_clip = RGB_nl

    #Post Gamut Mapping
    if do_map:
        print("Post Gamut Mapping enabled, use --no-map to disable...")
        JCh_reshape = JCh.reshape(shape)
        JC = JCh_reshape[..., 0] * JCh_reshape[..., 1]
        # Normalize JC to [0, 1] range
        normalized_JC = (JC - np.min(JC)) / (np.max(JC) - np.min(JC))
        normalized_JC = np.expand_dims(normalized_JC, axis=-1)  # Shape becomes (512, 512, 1)
        RGB_prime = RGB_i*normalized_JC + RGB_clip*(1-normalized_JC)
    else:
        print("Post Gamut Mapping disabled, use -m, --map to enable...")
        RGB_prime = RGB_clip

    #print(np.max(RGB_prime))
    
    
    print("Simulate nightlight for original image...")
    without_enhancement = temp_by_channel_gain_cont(RGB_i, temperature)
    print("Simulate nightlight for enhanced image...")
    with_enhancement = temp_by_channel_gain_cont(RGB_prime, temperature)
    
    # Create a figure with two subplots side by side
    plt.figure(figsize=(10, 5))  # Adjust the figure size

    # Show the first image
    plt.subplot(1, 3, 1)  # 1 row, 2 columns, first subplot
    plt.imshow(without_enhancement)
    plt.axis('off')  # Hide axes
    plt.title("without enhancement")  # Optional title

    plt.subplot(1, 3, 2)  # 1 row, 2 columns, first subplot
    plt.imshow(RGB_i)
    plt.axis('off')  # Hide axes
    plt.title("original image")  # Optional title

    plt.subplot(1, 3, 3)  # 1 row, 2 columns, second subplot
    plt.imshow(with_enhancement)
    plt.axis('off')  # Hide axes
    plt.title("with enhancement")  # Optional title

    # Display the images
    plt.tight_layout()  # Adjust spacing
    plt.show()
    