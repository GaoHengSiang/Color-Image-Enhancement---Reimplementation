import numpy as np
import cv2
import matplotlib.pyplot as plt

def device_rgb_to_xyz(image, M, gamma_r, gamma_g, gamma_b) -> np.ndarray:
    """
    Transforms an RGB image using the formula [x y z] = M [R^gamma_r G^gamma_g B^gamma_b].
    
    Args:
        image (numpy.ndarray): Input image as a 3D numpy array (H x W x 3) in RGB format.

        M (numpy.ndarray): 3x3 transformation matrix. (device characteristic)

        gamma_r (float): Gamma correction for the red channel. (device characteristic)

        gamma_g (float): Gamma correction for the green channel. (device characteristic)

        gamma_b (float): Gamma correction for the blue channel. (device characteristic)

    Returns:
        numpy.ndarray: Transformed image (H x W x 3).
    """
    # Apply gamma correction to each channel
    image_gamma = np.zeros_like(image, dtype=np.float32)
    h ,w, _ = image.shape

    image_gamma[..., 0] = image[..., 0] ** gamma_r  # R^gamma_r
    image_gamma[..., 1] = image[..., 1] ** gamma_g  # G^gamma_g
    image_gamma[..., 2] = image[..., 2] ** gamma_b  # B^gamma_b

    # Apply the matrix transformation
    transformed = np.dot(M, image_gamma.reshape(-1, 3).T).T

    # Reshape back to the original image dimensions
    return transformed.reshape(h, w, 3)

def device_xyz_to_rgb(image, M, gamma_r, gamma_g, gamma_b) -> np.ndarray:
    """
    Performs the inverse operation of device_rgb_to_xyz
    
    Args:
        image (numpy.ndarray): Input image as a 3D numpy array (H x W x 3) in XYZ format.

        M (numpy.ndarray): 3x3 transformation matrix. (device characteristic)

        gamma_r (float): Gamma correction for the red channel. (device characteristic)

        gamma_g (float): Gamma correction for the green channel. (device characteristic)

        gamma_b (float): Gamma correction for the blue channel. (device characteristic)

    Returns:
        numpy.ndarray: Transformed image (H x W x 3).
    """
    h, w, _ = image.shape
    M_inverse = np.linalg.inv(M)
    image_inverse = np.dot(M_inverse, image.reshape(-1, 3).T).T
    image_inverse = image_inverse.reshape(h, w, 3)
    image_inverse[..., 0] = image_inverse[..., 0] ** (1/gamma_r)  
    image_inverse[..., 1] = image_inverse[..., 1] ** (1/gamma_g)  
    image_inverse[..., 2] = image_inverse[..., 2] ** (1/gamma_b)

    # Reshape back to the original image dimensions
    return image_inverse.astype(np.uint8)




def simulated_low_backlight (image: np.ndarray, M_f, gamma_rf, gamma_gf, gamma_bf) -> np.ndarray: 
    """
    Given RGB image and the viewing device's characteristics, simulates what it looks like on a low backlight display
    
    Args:
        image (numpy.ndarray): Input image as a 3D numpy array (H x W x 3) in RGB format.

        Mf (numpy.ndarray): 3x3 transformation matrix. (device characteristic)

        gamma_rf (float): Gamma correction for the red channel. (device characteristic)

        gamma_gf (float): Gamma correction for the green channel. (device characteristic)

        gamma_bf (float): Gamma correction for the blue channel. (device characteristic)

    Returns:
        numpy.ndarray: Transformed image (H x W x 3).
    """
    # Virtual low backlight display characteristics =================================
    gamma_rl, gamma_gl, gamma_bl = 2.2212, 2.1044, 2.1835
    M_l = np.array([[4.61, 3.35, 1.78],
                    [2.48, 7.16, 0.79],
                    [0.28, 1.93, 8.93]])
    
    #=========================================================
    image_gamma = np.zeros_like(image, dtype=np.float32)
    h ,w, _ = image.shape

    #transform into tristimulus value given the low backlight display

    image_gamma[..., 0] = image[..., 0] ** gamma_rl  
    image_gamma[..., 1] = image[..., 1] ** gamma_gl  
    image_gamma[..., 2] = image[..., 2] ** gamma_bl  
    # Apply the matrix transformation
    transformed = np.dot(M_l, image_gamma.reshape(-1, 3).T)

    #=========================================================
    # Inverse transform with the viewing device's characteristic to simulate the effect
    M_f_inverse = np.linalg.inv(M_f)
    image_inverse = np.dot(M_f_inverse, transformed).T
    image_inverse = image_inverse.reshape(h, w, 3)
    image_inverse[..., 0] = image_inverse[..., 0] ** (1/gamma_rf)  
    image_inverse[..., 1] = image_inverse[..., 1] ** (1/gamma_gf)  
    image_inverse[..., 2] = image_inverse[..., 2] ** (1/gamma_bf)

    # Reshape back to the original image dimensions
    return image_inverse.astype(np.uint8)

if __name__ == "__main__":
    # Read the image
    image_bgr = cv2.imread("Lenna.png")

    # Check if the image was successfully loaded
    if image_bgr is None:
        raise FileNotFoundError("Image not found at the specified path!")

    # Convert the image from BGR (OpenCV default) to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Display the image dimensions
    print("Image shape (H, W, C):", image_rgb.shape)

    # Sample image
    
    #TIPS ============================================
    #when declaring array                            #
    #outer --> inner  <==> first axis --> last axis  #
    #                                                #
    #when reshaping/ravelling                        #
    #last axis --> first axis                        #
    #=================================================

    #image = np.array([[[255, 0, 0], [128, 128, 128], [0, 0, 255]],  #[red green blue], [red green blue] ...
    #                [[0, 255, 0], [255, 255, 0], [0, 255, 255]],
    #                [[0, 0, 255], [128, 0, 128], [255, 255, 255]]], dtype=np.uint8)

    # Device Characteristics =================================
    #full light
    gamma_rf, gamma_gf, gamma_bf = 2.4767, 2.4286, 2.3792
    M_f = np.array([[95.57,  64.67,  33.01],
                    [49.49, 137.29,  14.76],
                    [ 0.44,  27.21, 169.83]])

    #low light
    gamma_rl, gamma_gl, gamma_bl = 2.2212, 2.1044, 2.1835
    M_l = np.array([[4.61, 3.35, 1.78],
                    [2.48, 7.16, 0.79],
                    [0.28, 1.93, 8.93]])
    #=========================================================


    #Process the image
    #transformed_image = device_characteristic_modelling(image, M, gamma_r, gamma_g, gamma_b)
    transformed_image = simulated_low_backlight(image_rgb, M_f, gamma_rf, gamma_gf, gamma_bf)

    # Create a figure with two subplots side by side
    plt.figure(figsize=(10, 5))  # Adjust the figure size

    # Show the first image
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axes
    plt.title("full light")  # Optional title

    # Show the second image
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.imshow(transformed_image)
    plt.axis('off')  # Hide axes
    plt.title("low light")  # Optional title

    # Display the images
    plt.tight_layout()  # Adjust spacing
    plt.show()