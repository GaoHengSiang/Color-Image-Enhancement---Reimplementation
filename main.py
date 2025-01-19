import os
import numpy as np 
from ciecam02 import CIECAM02 
from parameters import constants, matrices, colordata 
from night_mode_simulation import convert_xyz_to_srgb, save_image
from preprocessing import preprocess_image 

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
def attribute_boosting(
        ciecam_image: np.ndarray, 
        lightness: bool=False,
        chroma: bool=False,
        boost_factor: float=1.0) -> np.ndarray:
    """Boost the lightness and/or chroma attributes of the CIECAM02 image

    Args:
        ciecam_image (np.ndarray): CIECAM02 image (JCh n x 3 np.ndarray).
        lightness (bool, optional): Lightness boost. Defaults to False.
        chroma (bool, optional): Chroma boost. Defaults to False.
        boost_factor (float, optional): Reduction factor Defaults to 1.0.

    Returns:
        np.ndarray: Boosted CIECAM02 image (JCh n x 3 np.ndarray).
    """
    # Extract J, C, and h
    J, C, h = ciecam_image[:, 0], ciecam_image[:, 1], ciecam_image[:, 2]

    # Dynamic scaling factors based on reduction_factor
    if lightness:
        lightness_boost_factor = 1.0 + 0.1 * boost_factor
        J = J + (lightness_boost_factor - 1.0) * (100 - J)
        J = np.clip(J, 0, 100)
    
    if chroma:
        chroma_boost_factor = 1.0 + 0.2 * boost_factor
        hue_weights = np.where((h > 200) & (h < 300), 1.2, 1.0)  # Boost blue-cyan hues
        C = C + (chroma_boost_factor - 1.0) * (100 - C) * hue_weights
        C = np.clip(C, 0, 100)
    
    # Reconstruct CIECAM02 array
    ciecam_image = np.stack([J, C, h], axis=1)

    return ciecam_image

# =============================================================================
def main(num_images: int=4, boosting: bool=False):
    """Main function to batch process images.

    Args:
        num_images (int): Number of images to process. Default is 4.
        boosting (bool): Flag to enable/disable boosting. Default is False.
    """
    # Define directories 
    original_dir = "./srgb_images" # Directory containing original images
    blue_light_dir = "./blue_light" # Directory containing blue light images
    output_dir = "./output_images" # Directory to save enhanced images
    boosted_dir = "./boosted_images" # Directory to save enhanced images with chroma boost

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(boosted_dir):
        os.makedirs(boosted_dir, exist_ok=True)

    # Initialise the CIECAM02 model
    model = CIECAM02(
            constants=constants, 
            matrices=matrices, 
            colordata=colordata)
    
    for i in range(1, num_images + 1): 
        # Define paths for the blue light-reduced and enhanced images
        original_filename = f"srgb_images_{i}.jpg"
        blue_light_filename = f"blue_light_reduced_{i}.jpg"
        output_filename = f"enhanced_image_{i}.jpg" 
        boosted_filename = f"boosted_image_{i}.jpg"

        original_path = os.path.join(original_dir, original_filename) 
        blue_light_path = os.path.join(blue_light_dir, blue_light_filename)
        output_path = os.path.join(output_dir, output_filename) 
        boosted_path = os.path.join(boosted_dir, boosted_filename) 

        if not os.path.exists(original_path):
            print(f"Image {original_path} not found in {original_dir}. Skipping...") 
            continue 

        print(f"Processing {original_filename}...")

        # Find the reference white point for the original image
        xyz_image, original_white = find_reference_white(
            image_path=original_path)
        constants["whitepoint"]["original"] = original_white

        # Find the reference white point for the blue light-reduced image
        _, blue_white = find_reference_white(
            image_path=blue_light_path)

        # Add this reference white point to the constants dictionary
        constants["whitepoint"]["blue_light"] = blue_white

        # CIECAM02 Forward Model 
        model.configure(
            white="original", 
            surround="average",
            light="default",
            bg="default")
        original_shape = xyz_image.shape
        ciecam_image = model.xyz_to_ciecam02(xyz_image.reshape(-1, 3))

        # # Extract J, C, and h
        # J, C, h = ciecam_image[:, 0], ciecam_image[:, 1], ciecam_image[:, 2]

        # # Dynamic scaling factors based on reduction_factor
        # lightness_boost_factor = 1.0 + 0.1 * 0.2
        # chroma_boost_factor = 1.0 + 0.2 * 1.0

        # # # Enhance lightness (tone-dependent adjustment)
        # # J = J + (lightness_boost_factor - 1.0) * (100 - J)
        # # J = np.clip(J, 0, 100)

        # # Enhance chroma (hue-weighted adjustment)
        # hue_weights = np.where((h > 200) & (h < 300), 1.2, 1.0)  # Boost blue-cyan hues
        # C = C + (chroma_boost_factor - 1.0) * (100 - C) * hue_weights
        # C = np.clip(C, 0, 100)

        # # Reconstruct CIECAM02 array
        # ciecam_image = np.stack([J, C, h], axis=1)

        # Boost lightness and chroma attributes
        if boosting:
            ciecam_image = attribute_boosting(
                ciecam_image, 
                lightness=False, 
                chroma=True, 
                boost_factor=1.0)

        # Inverse Model 
        model.configure(
            white="blue_light", 
            surround="average",
            light="default",
            bg="default")
        enhanced_xyz = model.inverse_model(ciecam_image).reshape(original_shape)

        # Convert enhanced XYZ to sRGB
        enhanced_srgb = convert_xyz_to_srgb(enhanced_xyz)
        enhanced_srgb = np.clip(enhanced_srgb, 0, 1)

        # Save the enhanced image
        if boosting:
            save_image(enhanced_srgb, boosted_path)
            print(f"Boosted image saved at {boosted_path}")
        else:
            save_image(enhanced_srgb, output_path)
            print(f"Enhanced image saved at {output_path}")
        
# =============================================================================
if __name__ == "__main__":
    main(num_images=8, boosting=False)  # You can change the number of images to process here
