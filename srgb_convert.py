import os 
from PIL import Image, ImageCms 

# =============================================================================
def is_image_srgb(image_path: str) -> bool: 
    """Check if the image has an sRGB color profile. 

    Args:
        image_path (_type_): Path to the input image. 

    Returns:
        bool: True if the image is in sRGB, False otherwise. 
    """
    try: 
        image = Image.open(image_path) 
        # Check for the presence of an ICC profile 
        icc_profile = image.info.get("icc_profile") 
        if icc_profile:
            return "sRGB" in icc_profile 
        return False 
    except Exception as e:
        print(f"Error checking sRGB profile for {image_path}: {e}") 
        return False 

# =============================================================================
def convert_to_srgb_with_icc(image_path: str, output_path: str) -> None:
    """Convert an image to sRGb color space with explicit ICC handling. 

    Args:
        image_path (str): Path to the input image 
        output_path (str): Path to the output image 
    """
    try: 
        # Open the image 
        image = Image.open(image_path)

        # # Handle RGBA images by converting them to RGB with a white background
        # if image.mode == "RGBA":
        #     print(f"Converting RGBA image to RGB for {image_path}")
        #     background = Image.new("RGB", image.size, (255, 255, 255))  # White background
        #     image = Image.alpha_composite(background, image.convert("RGBA"))
        
        # Check for an ICC profile 
        icc_profile = image.info.get("icc_profile") 
        if icc_profile:
            # Define the sRGB profile 
            srgb_profile = ImageCms.createProfile("sRGB") 
            # Create an ICC transform to sRGB 
            transform = ImageCms.buildTransformFromOpenProfiles(
                ImageCms.ImageCmsProfile(icc_profile),
                srgb_profile, 
                "RGB", 
                "RGB" 
            )
            image = ImageCms.applyTransform(image, transform) 
        
        # Save the converted image 
        image.save(output_path, "JPEG", quality=95)
        print(f"Image successfully converted to sRGB and saved at {output_path}") 
    except Exception as e:
        print(f"Error converting image to sRGB for {image_path}: {e}")

# =============================================================================
def batch_process_images(
        input_dir: str, 
        output_dir: str, 
        start_num: int=1, 
        end_num: int=10) -> None: 
    """Batch process images in a directory by checking and converting them to sRGB. 

    Args:
        input_dir (str): Directory containing the input images 
        output_dir (str): Directory to save the processed images. 
        start_num (int, optional): Starting index of the images to process.   
            Defaults to 1.
        end_num (int, optional): Ending index of the images to process.
            Defaults to 10.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

    for num in range(start_num, end_num + 1): 
        input_filename = f"test_image_{num}.png" 
        output_filename = f"srgb_images_{num}.jpg" 
        input_path = os.path.join(input_dir, input_filename) 
        output_path = os.path.join(output_dir, output_filename) 

        if not os.path.exists(input_path):
            print(f"Image {input_path} not found in {input_dir}. Skipping...") 
            continue 

        print(f"Processing {input_filename}...") 

        # Check if the image is in sRGB 
        if is_image_srgb(input_path): 
            print(f"{input_filename} is already sRGB. Copying to output directory.") 
            Image.open(input_path).save(output_path, "JPEG", quality=95) 
        else:
            print(f"{input_filename} is not in sRGB. Converting...") 
            convert_to_srgb_with_icc(input_path, output_path)
    print("Batch processing complete.") 

# =============================================================================
if __name__ == "__main__":
    input_dir = "./images" 
    output_dir = "./srgb_images" 

    # Define the range of images to process 
    start_idx = 1 
    end_idx = 8

    # Run batch processing 
    batch_process_images(input_dir, output_dir, start_idx, end_idx)