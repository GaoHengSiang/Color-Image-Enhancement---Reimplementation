import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
def collate_images(folder_name: str, ylabel: str):
    # Set the target size for the square images
    TARGET_SIZE = (256, 256)  # You can adjust this size as needed

    # Get list of image files from each folder
    # images_list = [] 
    images_list = sorted(
        [f for f in os.listdir(folder_name) if f.endswith(('.png', '.jpg', '.jpeg'))])
    # images_dict[folder_name] = sorted(
    #     [f for f in os.listdir(folder_name) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # Calculate the number of columns based on the maximum number of images in any folder
    num_cols = len(images_list)  # max([len(imgs) for imgs in images_list])

    # Create figure with specified size
    fig = plt.figure(figsize=(15, 4))
    
    # Create a grid of subplots
    grid = plt.GridSpec(1, num_cols, figure=fig)
    
    # Loop through folders and images
    for col, img_file in enumerate(images_list):
        img_path = os.path.join(folder_name, img_file)
        img = Image.open(img_path)
        # Resize image to square size while maintaining aspect ratio
        img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)
        # Create new square image with white background
        squared_img = Image.new('RGB', TARGET_SIZE, (255, 255, 255))
        # Paste the resized image in the center
        offset = ((TARGET_SIZE[0] - img.size[0]) // 2,
                 (TARGET_SIZE[1] - img.size[1]) // 2)
        squared_img.paste(img, offset)
        img_array = np.array(squared_img)
        
        # Create subplot
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(img_array)
        ax.axis('off')

    # Add shared y-label
    fig.supylabel(ylabel, size=12, rotation=90)
    plt.tight_layout()
    plt.show()

# =============================================================================
if __name__ =="__main__":
    collate_images(folder_name="./srgb_images", ylabel="Original")
    collate_images(folder_name="./blue_light", ylabel="Blue Light Reduced")
    collate_images(folder_name="./output_images", ylabel="Enhanced")
    collate_images(folder_name="./boosted_images", ylabel="Chroma-boosted")

