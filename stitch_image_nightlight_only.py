from PIL import Image
import os


def resize_image_to_fixed_height(image_path, target_height):
    """
    Resize an image to a fixed height while maintaining its aspect ratio.
    """
    img = Image.open(image_path)
    aspect_ratio = img.width / img.height
    new_width = int(target_height * aspect_ratio)  # Calculate the width while maintaining aspect ratio
    img_resized = img.resize((new_width, target_height))  # Resize with anti-aliasing
    return img_resized


def stitch_images_horizontally(image_paths, target_height, output_path):
    """
    Stitch a list of images horizontally after resizing them to the same height.
    """
    resized_images = [resize_image_to_fixed_height(img_path, target_height) for img_path in image_paths]

    # Calculate the total width and height of the stitched image
    total_width = sum(img.width for img in resized_images)
    max_height = max(img.height for img in resized_images)  # Should be the same as `target_height`

    # Create a blank canvas for the stitched image
    stitched_image = Image.new("RGB", (total_width, max_height))

    # Paste each image onto the canvas
    x_offset = 0
    for img in resized_images:
        stitched_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the stitched image
    stitched_image.save(output_path)


if __name__ == "__main__":

    # Directory paths and image names
    result_path = "./results/"
    color_images = ["baboon.png", "fruits.png", "HappyFish.jpg", "Lenna.png", "nave.jpg",
                    "parrot.jfif", "parrot2.jpg", "peppers.png", "rosette.jpg", "tulips.png"]
    row_dirs = ["original/", "nightlight/100/vanilla/", 
                "nightlight/100/no_map/no_comp/", "nightlight/100/map/no_comp/"]

    row_paths = [result_path + row_dir for row_dir in row_dirs]
    image_paths = [[row_path + color_image for color_image in color_images] for row_path in row_paths]
    image_paths[1:] = [[image + ".png" for image in row] for row in image_paths[1:]]  # Adjusting file extensions

    output_dir = "./stitched_rows/"
    os.makedirs(output_dir, exist_ok=True)

    # Fixed height for all images
    fixed_height = 200  # Adjust as needed

    # Stitch each row and save as a separate image
    for row_index, row_images in enumerate(image_paths):
        output_path = os.path.join(output_dir, f"stitched_row_{row_index + 1}.png")
        stitch_images_horizontally(row_images, fixed_height, output_path)
        print(f"Saved row {row_index + 1} to {output_path}")
