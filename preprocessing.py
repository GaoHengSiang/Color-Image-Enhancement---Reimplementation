import numpy as np 
from PIL import Image 

# ==============================================================================
def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess the input image for color appearance modeling. 

    This function converts an input image from its default RGB format to the CIE XYZ color space for further processing in the color enhancement workflow. 

    Args:
        image_path (str): Path to the input image file. 

    Returns:
        np.ndarray: The processed image in the CIE XYZ color space, normalized to the range [0, 1]. 

    Raises:
        FileNotFoundError: If the input image file does not exist. 
        ValueError: If the input image cannot be processed. 

    Notes: 
        The conversion from sRGB to XYZ follows the standard sRGB to XYZ transformation matrix as defined by the International Color Consortium (ICC). 
    """
    # sRGB to XYZ transformation matrix
    sRGB_to_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])
    
    try: 
        # Load the image and convert to NumPy array 
        image = Image.open(image_path).convert("RGB") 
        rgb_image = np.array(image, dtype=np.float32) 

        # Normalize the RGB image to the range [0, 1]
        rgb_image /= 255.0

        # Convert sRGB to linear RGB 
        def gamme_decode(value):
            return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
        
        linear_rgb = gamme_decode(rgb_image)

        # Convert linear RGB to XYZ
        xyz_image = np.dot(linear_rgb, sRGB_to_XYZ.T)

        # Clamp XYZ values to [0, 1] to handle out-of-range outputs
        xyz_image = np.clip(xyz_image, 0, 1)

        return xyz_image 
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Input image file '{image_path}' not found.")
    except Exception as e:
        raise ValueError(f"Error processing input image: {str(e)}")
    
# ==============================================================================
def test_preprocessing(image_path: str):
    try:
        # Run preprocessing 
        xyz_image = preprocess_image(image_path) 

        # Output some details about the processed image 
        print(f"Processed Image Shape: {xyz_image.shape}")
        print(f"Preprocessed Image Sample (Top-Left Pixel): {xyz_image[0,0]}") \
        
        # Validate the range of XYZ values 
        if np.any(xyz_image < 0) or np.any(xyz_image > 1):
            print("Warning: XYZ values are out of range [0, 1].") 
        else:
            print("Success: All XYZ values are within the range [0, 1].") 
            
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.") 
    except Exception as e:
        print(f"An error occurred: {e}") 


# ==============================================================================
if __name__ ==  "__main__": 
    # Test the image preprocessing function
    for i in range(1, 5):
        image_path = f"./srgb_images/srgb_images_{i}.jpg"
        print(f"Processing image: {image_path}")
        test_preprocessing(image_path)
        print("\n")