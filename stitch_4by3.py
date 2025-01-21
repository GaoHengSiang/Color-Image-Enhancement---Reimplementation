import cv2
import numpy as np
from pathlib import Path

def stitch_modes(image: str, paths: list):

    # Create full paths for the row
    row = [path + image for path in paths]

    # Append ".png" to all elements except the first one
    row[1:] = [path + ".png" for path in row[1:]]

    img_row = [cv2.imread(path) for path in row]

    collage = np.hstack(img_row)
    return collage

def row_stitch(color_images: list, paths: list, output_path: str):
    #create the target directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    img_row = []
    for _img in color_images:
        # Create full paths for the row
        row = [path + _img for path in paths]

        # Append ".png" to all elements except the first one
        row[1:] = [path + ".png" for path in row[1:]]

        img_row = [cv2.imread(path) for path in row]

        collage = np.hstack(img_row)
        cv2.imshow("Stitched Image", collage)
        cv2.imwrite(output_path+_img+".png", collage)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

if __name__ == "__main__":
    result_path = "./results/"

    original_path = result_path+"original/"

    nightlightstrengths = ["25/", "50/", "75/", "100/"]
    nightlightmodes = ["vanilla/", "no_map/no_comp/", "map/no_comp/"]
    color_images = ["tulips.png"]
    
    output_path = "./results/4by3/"

    nightlight_paths = [result_path + "nightlight/" + nightlightstrengths[1] + mode for mode in nightlightmodes]
    nightlight_paths = [original_path] + nightlight_paths  # Add original_path at the start
    
    #create the target directory
    stren_path = output_path+"strength/"
    Path(stren_path).mkdir(parents=True, exist_ok=True)

    #different images
    for _img in color_images:
        #vary the strength
        img_rows = []
        for stren in nightlightstrengths:
            nightlight_paths = [result_path + "nightlight/" + stren + mode for mode in nightlightmodes]
            nightlight_paths = [original_path] + nightlight_paths  # Add original_path at the start
            img_row = stitch_modes(image=_img, paths=nightlight_paths)
            img_rows.append(img_row)
        collage = np.vstack(img_rows)
        cv2.imshow("Stitched Image", collage)
        cv2.imwrite(stren_path+_img+".png", collage)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()