import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def device_rgb_to_xyz(image, M, gamma_r, gamma_g, gamma_b) -> np.ndarray:
    """
    Transforms an RGB image using the formula [x y z] = M [R^gamma_r G^gamma_g B^gamma_b].
    
    Args:
        image (numpy.ndarray): Input image as a 3D numpy array (H x W x 3) in RGB format, normalized to [0, 1].

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

    # Assert all values are in the range [0, 1]
    assert np.all(image >= 0), "Some values are less than 0!"
    assert np.all(image <= 1), "Some values are greater than 1!"

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


    # Assert all values are in the range [0, inf)
    #assert np.all(image_inverse >= 0), "Some values are less than 0!"
    if np.any(image_inverse < 0):
        print ("clipping negative intermediate values during XYZ to RGB conversion")
        image_inverse = np.clip(image_inverse, a_min = 0, a_max = None)

    image_inverse[..., 0] = image_inverse[..., 0] ** (1/gamma_r)  
    image_inverse[..., 1] = image_inverse[..., 1] ** (1/gamma_g)  
    image_inverse[..., 2] = image_inverse[..., 2] ** (1/gamma_b)

    if np.any((image_inverse < 0) | (image_inverse > 1)):
        image_inverse = np.clip(image_inverse, 0, 1)
        print ("XYZ to RGB conversion results in out of range values, consider post gamut mapping")

    return image_inverse

def simulated_low_backlight (image_rgb: np.ndarray, M, gamma_r, gamma_g, gamma_b) -> np.ndarray: 
    """
      Given RGB image and the viewing device's characteristics, simulates what it looks like on a low backlight display
    
    Args:
        image (numpy.ndarray): Input image as a 3D numpy array (H x W x 3) in RGB format, normalized to 0, 1.

        M (np.ndarray): target device characteristic, use M_l in this case 

        gamma_r (float): target device characteristic, use gamma_rl in this case 

        gamma_g (float): target device characteristic, use gamma_gl in this case 

        gamma_b (float): target device characteristic, use gamma_bl in this case 

    Returns:
        numpy.ndarray: Transformed image (H x W x 3).
    """
    # Virtual low backlight display characteristics =================================
    #gamma_rl, gamma_gl, gamma_bl = 2.2212, 2.1044, 2.1835
    #M_l = np.array([[4.61, 3.35, 1.78],
    #                [2.48, 7.16, 0.79],
    #                [0.28, 1.93, 8.93]])
    #
    #=========================================================
    # Assert all values are in the range [0, 1]
    assert np.all(image_rgb >= 0), "Some values are less than 0!"
    assert np.all(image_rgb <= 1), "Some values are greater than 1!"

    #transform into tristimulus value given the low backlight display

    image_xyz = device_rgb_to_xyz(image_rgb, M, gamma_r, gamma_g, gamma_b)
    image_inverse = device_xyz_to_rgb(image_xyz, M_f, gamma_rf, gamma_gf, gamma_bf)
    #image_gamma[..., 0] = image[..., 0] ** gamma_rl  
    #image_gamma[..., 1] = image[..., 1] ** gamma_gl  
    #image_gamma[..., 2] = image[..., 2] ** gamma_bl  
    # Apply the matrix transformation
    #transformed = np.dot(M_l, image_gamma.reshape(-1, 3).T)

    #=========================================================
    # Inverse transform with the viewing device's characteristic to simulate the effect
    #M_f_inverse = np.linalg.inv(M_f)
    #image_inverse = np.dot(M_f_inverse, transformed).T
    #image_inverse = image_inverse.reshape(h, w, 3)
    #image_inverse[..., 0] = image_inverse[..., 0] ** (1/gamma_rf)  
    #image_inverse[..., 1] = image_inverse[..., 1] ** (1/gamma_gf)  
    #image_inverse[..., 2] = image_inverse[..., 2] ** (1/gamma_bf)

    # Reshape back to the original image dimensions
    return image_inverse

def post_gamut_mapping(original_rgb: np.ndarray, enhanced_xyz, JC: np.ndarray) -> np.ndarray:
    """
      The enhanced tristimulus (XYZ) values may not map into valid RGB space, hence it is necessary
    to perform post-gamut-mapping.
      Alleviate loss of details, we implement soft clipping with weighted average

    Args:
        original_rgb (H x W x 3 np.ndarray): original RGB image

        enhanced_xyz (H x W x 3 np.ndarray): X Y Z tristimulus image

        JC (H x W np.ndarray): J*C, The coefficient of weighted average for each pixel

    Returns:
        numpy.ndarray: Transformed image (H x W x 3).
    """ 
    assert np.all(original_rgb >= 0), "rgb should be normalized to [0, 1]"
    assert np.all(original_rgb <= 1), "rgb should be normalized to [0, 1]"
    assert np.all(JC >= 0), "JC should be normalized to [0, 1]"
    assert np.all(JC <= 1), "JC should be normalized to [0, 1]"

    clipped_rgb = device_xyz_to_rgb(enhanced_xyz, M_l, gamma_rl, gamma_gl, gamma_bl)
      # Perform weighted average
    #result = cv2.addWeighted(original_rgb, JC, clipped_rgb, 1-JC, gamma = 0)
    JC = np.expand_dims(JC, axis=-1)  # Shape becomes (512, 512, 1)
    result = original_rgb*JC + clipped_rgb*(1-JC)
    return result


if __name__ == "__main__":
    # Read the image
    image_bgr = cv2.imread("./test_images/fruits.png")

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

    #Process the image
    image_rgb = image_rgb/255.0
    transformed_image1 = simulated_low_backlight(image_rgb, M_l, gamma_rl, gamma_gl, gamma_bl)

    #test post gamut mapping
    #1. show tristimulus value of larger range
    #_img = device_rgb_to_xyz(image_rgb, M_f, gamma_rf, gamma_gf, gamma_bf)
    #low backlight should not be able to handle full gamut, resulting in out of range RGB values
    #transformed_image2 = post_gamut_mapping(image_rgb, _img, 0.5) #return rgb values
    #to view the effect on normal screen,  we simulate low backlight (using M_f, gamma_xf as my devices characteristic)
    #transformed_image2 = simulated_low_backlight(transformed_image2, M_f, gamma_rf, gamma_gf, gamma_bf) 

    sample_white = np.ones((1, 1, 3), dtype = np.float32)
    W_f = device_rgb_to_xyz(sample_white, M_f, gamma_rf, gamma_gf, gamma_bf)
    W_l = device_rgb_to_xyz(sample_white, M_l, gamma_rl, gamma_gl, gamma_bl)
    print(f"W_f = {W_f}")
    print(f"W_l = {W_l}")

    # Create a figure with two subplots side by side
    plt.figure(figsize=(10, 5))  # Adjust the figure size

    # Show the first image
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axes
    plt.title("full light")  # Optional title

    # Show the second image
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.imshow(transformed_image1)
    plt.axis('off')  # Hide axes
    plt.title("low light")  # Optional title

    # Display the images
    plt.tight_layout()  # Adjust spacing
    plt.show()
