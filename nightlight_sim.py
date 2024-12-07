"""
This code is courtesy of:
https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html

Applies nightlight effect by multiplying the RGB channel with different gains
closely simulates the windows nightlight effect

"""



import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def apply_night_light_effect_cv2(image: np.ndarray, intensity=0.5):
    """
    Apply a Night Light effect to an image using OpenCV and display it with Matplotlib.

    Args:
        image_path (str): Path to the input image.
        intensity (float): Level of blue reduction (0.0 to 1.0). Higher means more reduction.
    """

    # Split channels
    r, g, b = cv2.split(image)

    # Reduce blue channel
    b = np.clip(b * (1 - intensity), 0, 255).astype(np.uint8)

    # Slightly boost red and green channels
    r = np.clip(r + 20 * intensity, 0, 255).astype(np.uint8)
    g = np.clip(g + 10 * intensity, 0, 255).astype(np.uint8)

    # Merge channels back
    result = cv2.merge((r, g, b))

    # Optionally reduce overall brightness
    result = np.clip(result * 0.9, 0, 255).astype(np.uint8)

    return result


def temp_by_channel_gain_tabl(image, temp):
    kelvin_table = {
        1000: (255, 56, 0),
        1500: (255, 109, 0),
        2000: (255, 137, 18),
        2500: (255, 161, 72),
        3000: (255, 180, 107),
        3500: (255, 196, 137),
        4000: (255, 209, 163),
        4500: (255, 219, 186),
        5000: (255, 228, 206),
        5500: (255, 236, 224),
        6000: (255, 243, 239),
        6500: (255, 249, 253),
        7000: (245, 243, 255),
        7500: (235, 238, 255),
        8000: (227, 233, 255),
        8500: (220, 229, 255),
        9000: (214, 225, 255),
        9500: (208, 222, 255),
        10000: (204, 219, 255)
    }

    # Get the RGB values for the desired color temperature
    r, g, b = kelvin_table[temp]

    # Normalize the RGB values
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # Apply the transformation to the input image (assuming it's a NumPy array)
    image = np.float32(image)  # Convert the image to float for scaling
    image[..., 0] *= r  # Apply scaling to Red channel
    image[..., 1] *= g  # Apply scaling to Green channel
    image[..., 2] *= b  # Apply scaling to Blue channel

    # Clip values to ensure they remain valid for uint8 (0-255)
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image
import numpy as np
import cv2

def temp_by_channel_gain_cont(image, temperature):
    """
    Adjusts the color temperature of an image based on a given Kelvin value.

    Args:
        image (numpy.ndarray): Input image in RGB format.
        temperature (float): Desired color temperature in Kelvin (1000 to 40000).

    Returns:
        numpy.ndarray: Color temperature-adjusted image.
    """
    # Normalize temperature
    temp = temperature / 100.0

    # Compute RGB scaling factors
    if temp <= 66:
        red = 255.0
        green = 99.4708025861 * np.log(temp) - 161.1195681661
        blue = 0.0 if temp <= 19 else 138.5177312231 * np.log(temp - 10) - 305.0447927307
    else:
        red = 329.698727446 * ((temp - 60) ** -0.1332047592)
        green = 288.1221695283 * ((temp - 60) ** -0.0755148492)
        blue = 255.0

    # Clamp RGB scaling factors to [0, 255]
    red = np.clip(red, 0, 255)
    green = np.clip(green, 0, 255)
    blue = np.clip(blue, 0, 255)

    # Normalize scaling factors to [0, 1] --> float
    red /= 255.0
    green /= 255.0
    blue /= 255.0

    # Scale image channels
    image = np.float32(image)  # Ensure image is float for scaling
    image[..., 0] *= red  # Scale Red channel
    image[..., 1] *= green  # Scale Green channel
    image[..., 2] *= blue  # Scale Blue channel

    # Clip values to [0, 255] and convert back to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image

def kelvin_to_scaling_factors(temperature):
    """
    Converts a color temperature in Kelvin to RGB scaling factors.
    """
    temp = temperature / 100.0

    # Compute RGB scaling factors
    if temp <= 66:
        red = 255.0
        green = 99.4708025861 * np.log(temp) - 161.1195681661
        blue = 0.0 if temp <= 19 else 138.5177312231 * np.log(temp - 10) - 305.0447927307
    else:
        red = 329.698727446 * ((temp - 60) ** -0.1332047592)
        green = 288.1221695283 * ((temp - 60) ** -0.0755148492)
        blue = 255.0

    # Clamp scaling factors
    red = max(0, min(255, red)) / 255.0
    green = max(0, min(255, green)) / 255.0
    blue = max(0, min(255, blue)) / 255.0

    return red, green, blue

def adjust_temperature(image, temperature):
    """
    Adjusts the color temperature of an image based on the Kelvin value.
    """
    red, green, blue = kelvin_to_scaling_factors(temperature)
    image = np.float32(image)  # Convert to float for scaling
    image[..., 0] *= red  # Scale Red channel
    image[..., 1] *= green  # Scale Green channel
    image[..., 2] *= blue  # Scale Blue channel
    return np.clip(image, 0, 255).astype(np.uint8)

def update_image(val):
    """
    Callback function to update the image based on the slider value.
    """
    strength = cv2.getTrackbarPos('Strength', 'Adjust Nightlight Strength')
    temp = map_windows_nightlight_slider_to_temp(strength)
    adjusted = temp_by_channel_gain_cont(image, temp)
    cv2.imshow('Adjust Nightlight Strength', cv2.cvtColor(adjusted, cv2.COLOR_RGB2BGR))

def map_windows_nightlight_slider_to_temp(slider_val: int) -> float:
    """
    mimics windows nightlight strength slider behavior
    assumption: linear relationship
    colormapping estimated with MY EYES
    """
    slider_val = max(0, min(slider_val, 100))
    #100 --> 992K
    low_temp = 992
    #0   --> 6700K
    high_temp = 6700
    equiv_temp = high_temp-slider_val*(high_temp-low_temp)/100.0
    return equiv_temp

if __name__ == "__main__":
    # Load the image
    image_path = "Desktop.png"
    white_patch = np.ones((500, 500, 3), dtype=np.uint8) * 255

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency

    temperature = 10000  # Set desired color temperature in Kelvin
    #result = temp_by_channel_gain_cont(image, temperature)


    # Create a window and trackbar
    cv2.namedWindow('Adjust Nightlight Strength')
    cv2.createTrackbar('Strength', 'Adjust Nightlight Strength', 100, 100, update_image)


    
    # Display the image using Matplotlib
    #plt.figure(figsize=(8, 6))
    #plt.imshow(result)
    #plt.axis("off")
    #plt.title("Night Light Effect")
    #plt.show()


    # Wait for user interaction
    cv2.waitKey(0)
    cv2.destroyAllWindows()