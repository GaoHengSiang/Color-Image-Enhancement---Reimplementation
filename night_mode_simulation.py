import os
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
from preprocessing import preprocess_image

# =============================================================================
def save_image(image_array: np.ndarray, output_path: str) -> None:
    """Save a Numpy array as an image.

    Args:
        image_array (np.ndarray): Image array
        output_path (str): Output path for the saved image
    """
    img = (image_array * 255).clip(0, 255).astype(np.uint8)  # Scale back to [0, 255]
    if len(img.shape) == 3 and img.shape[2] == 3:
        Image.fromarray(img, 'RGB').save(output_path)
    else:
        Image.fromarray(img, 'L').save(output_path)

# =============================================================================
def kelvin_to_scaling_factors_xyz(temperature: float) -> np.ndarray: 
    """Convert a color temperature in Kelvin to sclaing factors in the XYZ color space. 

    Args:
        temperature (float): Color temperature in Kelvin 

    Returns:
        np.ndarray: Scaling factor for X, Y, and Z channels. 
    """
    # Use the same scaling factor for RGB, but adapt them to XYZ 
    temp = temperature / 100.0 

    # Compute RGB scaling factors 
    if temp <= 65: 
        red = 255.0 
        green = 99.4708025861 * np.log(temp) - 161.1195681661
        blue = 0.0 if temp <= 19 else 138.5177312231 * np.log(temp - 10) - 305.0447927307
    else:
        red = 329.698727446 * np.power(temp - 60, -0.1332047592)
        green = 288.1221695283 * np.power(temp - 60, -0.0755148492)
        blue = 255.0

    # Normalize RGB scaling factors 
    red = np.clip(red, 0, 255) / 255.0 
    green = np.clip(green, 0, 255) / 255.0
    blue = np.clip(blue, 0, 255) / 255.0

    # Map RGB scaling to approximate XYZ scaling 
    rgb_to_xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    scaling_factors = np.dot([red, green, blue], rgb_to_xyz) 
    
    return scaling_factors 

# =============================================================================
def adjust_temperature_xyz(
        xyz_image: np.ndarray, 
        temperature: float) -> np.ndarray:
    """Adjust the color temperature of an image in the XYZ space. 

    Args:
        xyz_image (np.ndarray): Input image in the CIE XYZ color space. 
        temperature (float): Desired color temperature in Kelvin.

    Returns:
        np.ndarray: XYZ image with adjusted color temperature. 
    """
    # Get scaling factors for the XYZ space 
    scaling_factors = kelvin_to_scaling_factors_xyz(temperature) 

    # Apply scaling factors to the XYZ image 
    adjusted_xyz = xyz_image.copy() 
    adjusted_xyz[..., 0] *= scaling_factors[0]
    adjusted_xyz[..., 1] *= scaling_factors[1]
    adjusted_xyz[..., 2] *= scaling_factors[2]

    # Ensure values remain valid 
    return np.clip(adjusted_xyz, 0, 1)

# =============================================================================
# =============================================================================
def adjust_color_temperature(xyz: np.ndarray, kelvin_shift: int) -> np.ndarray:
    """Adjust color temperature of an image by modifying the white point.

    The funstion applies the scaling factors to the XYZ values channel-wise to alter the overall color balance. A warmer shift emphasies reds and reduces blues, while a cooler temperature shift increases blues and reduces reds.

    Args:
        xyz (np.ndarray): Image in the CIE XYZ color space
        kelvin_shift (int): Shift in color temperature. A positive shift 
            increases the color temperature, while a negative shift decreases it.

    Returns:
        np.ndarray: Adjusted image in the CIE XYZ color space
    """
    # Convert Kelvin to approximate RGB white point scaling factors
    if kelvin_shift > 0:
        scale = np.array([1.0 - 0.05 * kelvin_shift, 1.0 - 0.05 * kelvin_shift, 1.0 - 0.15 * kelvin_shift])
    else:
        scale = np.array([1.0 + 0.15 * -kelvin_shift, 1.0 + 0.05 * -kelvin_shift, 1.0 + 0.05 * -kelvin_shift])

    # Adjust XYZ by scaling relative to D65 reference white
    return xyz * scale

# =============================================================================
def reduce_blue_light(srgb_image_path: str, shifting_value: float) -> np.ndarray:
    """Simulate reduced blue light by adjusting the color temperature of an sRGB image. 

    This function applies a color temperature shift to an sRGB image to simulate reduced blue light exposure. 

    Args:
        srgb_image_path (str): Input image in the sRGB color space. 
        shifting_factor (float): Desired color temperature shift in Kelvin. 

    Returns:
        np.ndarray: The color temperature-adjusted image in the sRGB color space. 
    """
    # Convert sRGB to XYZ 
    xyz_image = preprocess_image(srgb_image_path)
    # print(xyz_image.shape)

    # Adjust color temperature in XYZ space 
    # adjusted_xyz = adjust_temperature_xyz(xyz_image, temperature)
    adjusted_xyz = adjust_color_temperature(
        xyz=xyz_image, 
        kelvin_shift=shifting_value) 
    # print(adjusted_xyz.shape)

    # Convert adjusted XYZ to sRGB 
    adjusted_srgb = convert_xyz_to_srgb(adjusted_xyz)
    # print(adjusted_srgb.shape)

    return np.clip(adjusted_srgb, 0, 1)

# =============================================================================
# =============================================================================
def convert_xyz_to_srgb(xyz_image: np.ndarray) -> np.ndarray:
    """Convert an image from the XYZ color space to the sRGB color space. 

    Args:
        xyz_image (np.ndarray): Input image in the CIE XYZ color space. 

    Returns:
        np.ndarray: Image in the sRGB color space. 
    """
    # XYZ to sRGB transformation matrix 
    xyz_to_srgb = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ])

    # Convert XYZ to sRGB 
    linear_srgb = np.dot(xyz_image, xyz_to_srgb.T)
    linear_srgb = np.clip(linear_srgb, 0, 1) 

    return np.where(
        linear_srgb <= 0.0031308, 
        12.92 * linear_srgb, 
        1.055 * np.power(linear_srgb, 1.0 / 2.4) - 0.055)

# =============================================================================
def show_temeprature_comparison_xyz(
        xyz_image: np.ndarray, 
        kelvin_shift: float) -> None:
    """Display a comparison of the original and color temperture-adjusted images. 

    Args:
        xyz_image (np.ndarray): Input image in the CIE XYZ color space. 
        kelvin_shift (float): Desired color temperature shift in Kelvin. 
    """
    # Adjust image temperature 
    # adjusted_xyz = adjust_temperature_xyz(xyz_image, temperature)
    adjusted_xyz = adjust_color_temperature(
        xyz_image, 
        kelvin_shift=kelvin_shift)

    # Convert XYZ to sRGB for visualisation 
    original_srgb = convert_xyz_to_srgb(xyz_image)
    original_srgb = (original_srgb * 255).clip(0, 255).astype(np.uint8)  # Scale back to [0, 255]
    adjusted_srgb = convert_xyz_to_srgb(adjusted_xyz)
    adjusted_srgb = (adjusted_srgb * 255).clip(0, 255).astype(np.uint8)

    # Display the original and adjusted images 
    plt.figure(figsize=(10, 5))

    # Original Image 
    plt.subplot(1, 2, 1)
    plt.imshow(original_srgb)
    plt.title('Original Image')
    plt.axis('off')

    # Temperature-adjusted Image
    plt.subplot(1, 2, 2)
    plt.imshow(adjusted_srgb)
    plt.title(f'Adjusted Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# =============================================================================
def batch_process_images(
        input_dir: str, 
        output_dir: str, 
        start_num: int=1, 
        end_num: int=10,
        shifting_value: float=3) -> None: 
    """Batch process images in a directory by checking and converting them to sRGB. 

    Args:
        input_dir (str): Directory containing the input images 
        output_dir (str): Directory to save the output images 
        start_num (int): Starting index for image filenames 
        end_num (int): Ending index for image filenames 
        shifting_value (float): Desired color temperature shift in Kelvin.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True) 

    for i in range(start_num, end_num + 1): 
        input_filename = f"srgb_images_{i}.jpg" 
        output_filename = f"blue_light_reduced_{i}.jpg" 
        input_path = os.path.join(input_dir, input_filename) 
        output_path = os.path.join(output_dir, output_filename) 

        if not os.path.exists(input_path):
            print(f"Image {input_path} not found in {input_dir}. Skipping...") 
            continue 

        print(f"Processing {input_filename}...")

        # Simulate reduced blue light
        blue_light_reduced_image = reduce_blue_light(
            srgb_image_path=input_path, 
            shifting_value=shifting_value) 
        save_image(blue_light_reduced_image, output_path)
        print(f"Blue light reduced image saved at {output_path}")

# =============================================================================
if __name__ == "__main__": 

    # # Load image 
    # image_path = "./srgb_images/srgb_images_5.jpg" 

    # try:
    #     # Prepare the image for processing
    #     xyz_image = preprocess_image(image_path)

    #     # Simulate blue-light reduction using temperature adjustment 
    #     # temperature = 4000  # Set desired color temperature in Kelvin (1500K-4000K for night mode)
    #     shifting_factor = -5
    #     show_temeprature_comparison_xyz(xyz_image, shifting_factor)
    # except FileNotFoundError:
    #     print(f"Error: Image file not found at {image_path}")
    # except ImportError:
    #     print("Error: Make sure preprocessing.py is in the same directory")

    # Batch process images in the sRGB directory
    input_dir = "./srgb_images"
    output_dir = "./blue_light"
    batch_process_images(
        input_dir, 
        output_dir, 
        start_num=1, 
        end_num=8,
        shifting_value=4)
