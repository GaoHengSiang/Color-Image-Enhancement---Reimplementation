import os
import numpy as np 
# from PIL import Image 
from ciecam02 import CIECAM02 
from parameters import constants, matrices, colordata 
# import matplotlib.pyplot as plt 
from night_mode_simulation import convert_xyz_to_srgb, save_image
from preprocessing import preprocess_image 

# =============================================================================
def enhance_color_appearance(model, image_path: str, reduction_factor: float = 1.0) -> np.ndarray:
    """Enhance the color appearance of an image under reduced blue-light conditions.

    Args:
        model (CIECAM02): CIECAM02 model instance.
        image_path (str): Path to the image (in the sRGB color space) to enhance.
        reduction_factor (float): Factor indicating the degree of blue-light reduction (0 < reduction_factor ≤ 1).

    Returns:
        np.ndarray: Enhanced image in the sRGB color space.
    """
    # Convert sRGB to XYZ
    xyz_image = preprocess_image(image_path)

    # Convert XYZ to CIECAM02 (JCh: n x 3 np.ndarray)
    original_shape = xyz_image.shape
    ciecam_image = model.xyz_to_ciecam02(xyz_image.reshape(-1, 3))

    # Extract J, C, and h
    J, C, h = ciecam_image[:, 0], ciecam_image[:, 1], ciecam_image[:, 2]

    # Dynamic scaling factors based on reduction_factor
    lightness_boost_factor = 1.0 + 0.1 * reduction_factor
    chroma_boost_factor = 1.0 + 0.2 * reduction_factor

    # Enhance lightness (tone-dependent adjustment)
    J = J + (lightness_boost_factor - 1.0) * (100 - J)
    J = np.clip(J, 0, 100)

    # Enhance chroma (hue-weighted adjustment)
    hue_weights = np.where((h > 200) & (h < 300), 1.2, 1.0)  # Boost blue-cyan hues
    C = C + (chroma_boost_factor - 1.0) * (100 - C) * hue_weights
    C = np.clip(C, 0, 100)

    # Reconstruct CIECAM02 array
    ciecam_image = np.stack([J, C, h], axis=1)

    # Convert enhanced CIECAM02 to XYZ
    enhanced_xyz = model.inverse_model(ciecam_image)

    # Convert enhanced XYZ to sRGB
    enhanced_srgb = convert_xyz_to_srgb(enhanced_xyz)

    # # Post gamut mapping
    # enhanced_xyz.reshape(original_shape)
    # JC = enhanced_xyz[..., 0] * enhanced_xyz[..., 1]
    # # Normalize JC to [0, 1] range
    # normalized_JC = (JC - np.min(JC)) / (np.max(JC) - np.min(JC))
    # normalized_JC = np.expand_dims(normalized_JC, axis=-1)  # Shape becomes (512, 512, 1)
    # RGB_prime = RGB_i*normalized_JC + RGB_clip*(1-normalized_JC)

    return np.clip(enhanced_srgb.reshape(original_shape), 0, 1)

# =============================================================================
def find_reference_white(image_path: str) -> np.ndarray:
    """Find the reference white point for the blue light-reduced image.

    Args:
        image_path (str): Path to the blue light-reduced image.

    Returns:
        np.ndarray: Reference white point in the XYZ color space.
    """
    # Convert sRGB to XYZ
    xyz_image = preprocess_image(image_path)

    # Calculate the mean XYZ values
    white_points = np.mean(xyz_image, axis=(0, 1))

    # # Normalize to Y = 1
    # white_points = white_points / white_points[1]

    return xyz_image, white_points

# =============================================================================
def main(num_images: int = 4):
    """Main function to batch process images.

    Args:
        num_images (int): Number of images to process. Default is 4.
    """
    # Define directories 
    input_dir = "./blue_light" # Directory containing blue light images
    output_dir = "./output_images" # Directory to save enhanced images

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Initialise the CIECAM02 model
    model = CIECAM02(
            constants=constants, 
            matrices=matrices, 
            colordata=colordata)
    
    for i in range(1, num_images + 1): 

        # Find the reference white point for the original image
        _, original_white = find_reference_white(
            image_path=f"./srgb_images/srgb_images_{i}.jpg")
        constants["whitepoint"]["original"] = original_white

        # Define paths for the blue light-reduced and enhanced images
        input_filename = f"blue_light_reduced_{i}.jpg"
        output_filename = f"enhanced_image_{i}.jpg" 
        input_path = os.path.join(input_dir, input_filename) 
        output_path = os.path.join(output_dir, output_filename) 

        if not os.path.exists(input_path):
            print(f"Image {input_path} not found in {input_dir}. Skipping...") 
            continue 

        print(f"Processing {input_filename}...")

        # Preprocess the blue light-reduced image from sRGB to XYZ
        xyz_image, blue_white = find_reference_white(image_path=input_path)

        # Add this reference white point to the constants dictionary
        constants["whitepoint"]["blue_light"] = blue_white

        # CIECAM02 Forward Model 
        model.configure(
            white="blue_light", 
            surround="average",
            light="default",
            bg="default")
        original_shape = xyz_image.shape
        ciecam_image = model.xyz_to_ciecam02(xyz_image.reshape(-1, 3))

        # Extract J, C, and h
        J, C, h = ciecam_image[:, 0], ciecam_image[:, 1], ciecam_image[:, 2]

        # Dynamic scaling factors based on reduction_factor
        lightness_boost_factor = 1.0 + 0.1 * 0.2
        chroma_boost_factor = 1.0 + 0.2 * 1.0

        # # Enhance lightness (tone-dependent adjustment)
        # J = J + (lightness_boost_factor - 1.0) * (100 - J)
        # J = np.clip(J, 0, 100)

        # Enhance chroma (hue-weighted adjustment)
        hue_weights = np.where((h > 200) & (h < 300), 1.2, 1.0)  # Boost blue-cyan hues
        C = C + (chroma_boost_factor - 1.0) * (100 - C) * hue_weights
        C = np.clip(C, 0, 100)

        # Reconstruct CIECAM02 array
        ciecam_image = np.stack([J, C, h], axis=1)

        # Inverse Model 
        model.configure(
            white="original", 
            surround="average",
            light="default",
            bg="default")
        enhanced_xyz = model.inverse_model(ciecam_image).reshape(original_shape)

        # Convert enhanced XYZ to sRGB
        enhanced_srgb = convert_xyz_to_srgb(enhanced_xyz)
        enhanced_srgb = np.clip(enhanced_srgb, 0, 1)

        # Save the enhanced image
        save_image(enhanced_srgb, output_path)
        print(f"Enhanced image saved at {output_path}")


# =============================================================================
if __name__ == "__main__":
    main(num_images=8)  # You can change the number of images to process here
