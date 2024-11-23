import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

# ========== CIECAM02 ========== #
def ciecam02(image_xyz: np.ndarray, La: float, Yb: float) -> np.ndarray:
    """
    Converts an XYZ image to simplified CIECAM02 attributes (lightness, chroma, hue).
    
    Args:
        image_xyz (np.ndarray): Input image in XYZ color space (H x W x 3).
        La (float): Luminance of the adapting field.
        Yb (float): Luminance of the background field.
    
    Returns:
        np.ndarray: Image with CIECAM02 attributes (lightness J, chroma C, hue h).
    """
    # Constants for the CIECAM02 model
    sR = 1.0  # Surround factor for 'average'
    n = Yb / La
    N_bb = N_cb = 0.725 * (1 / n) ** 0.2
    
    # Conversion from XYZ to LMS (Long, Medium, Short cones)
    M_CAT02 = np.array([[0.7328, 0.4296, -0.1624],
                        [-0.7036, 1.6975, 0.0061],
                        [0.0030, 0.0136, 0.9834]])
    
    h, w, _ = image_xyz.shape
    image_lms = np.dot(image_xyz.reshape(-1, 3), M_CAT02.T).reshape(h, w, 3)

    # Calculate the adapted cone responses
    D = sR * (1 - (1/3.6) * np.exp((-La - 42)/92))  # Discounting factor
    LMS_a = D * image_lms + (1 - D) * image_lms.mean(axis=(0, 1))  # Adaptation

    # Calculate lightness, chroma, and hue
    A = (2 * LMS_a[..., 0] + LMS_a[..., 1] + (1/20) * LMS_a[..., 2]) * N_bb
    J = 100 * (A / La) ** 0.5
    C = 100 * (np.abs(LMS_a[..., 0] - LMS_a[..., 2]) / La)
    h = np.degrees(np.arctan2(LMS_a[..., 1], LMS_a[..., 0]))

    # Stack the results in an array
    ciecam02_attributes = np.stack((J, C, h), axis=-1)
    return ciecam02_attributes

# ========== Inverse CIECAM02 ========== #
def inv_ciecam02(ciecam02_attributes: np.ndarray, La: float, Yb: float) -> np.ndarray:
    """
    Converts simplified CIECAM02 attributes (lightness, chroma, hue) back to the XYZ color space.
    
    Args:
        ciecam02_attributes (np.ndarray): CIECAM02 attributes (H x W x 3).
        La (float): Luminance of the adapting field.
        Yb (float): Luminance of the background field.
    
    Returns:
        np.ndarray: Reconstructed image in the XYZ color space.
    """
    # Constants
    sR = 1.0  # Average surround
    n = Yb / La
    N_bb = N_cb = 0.725 * (1 / n) ** 0.2
    
    # Extract lightness (J), chroma (C), and hue (h)
    J, C, h = ciecam02_attributes[..., 0], ciecam02_attributes[..., 1], ciecam02_attributes[..., 2]
    
    # Calculate achromatic response A from lightness J
    A = (J / 100) ** 2 * La
    
    # Calculate adapted LMS values
    LMS_a = np.zeros_like(ciecam02_attributes)
    LMS_a[..., 0] = (A / (2 * N_bb))
    LMS_a[..., 1] = C * np.sin(np.radians(h))
    LMS_a[..., 2] = C * np.cos(np.radians(h))
    
    # Convert LMS back to XYZ
    M_CAT02_inv = np.linalg.inv(
        np.array([
            [0.7328, 0.4296, -0.1624],
            [-0.7036, 1.6975, 0.0061],
            [0.0030, 0.0136, 0.9834]]))
    
    image_xyz = np.dot(
        LMS_a.reshape(-1, 3), M_CAT02_inv.T).reshape(ciecam02_attributes.shape)
    return np.clip(image_xyz, 0, 1)  # Clipping for valid range