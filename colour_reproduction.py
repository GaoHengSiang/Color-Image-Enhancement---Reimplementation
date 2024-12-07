import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

# ========== CIECAM02 ========== #
def calculate_viewing_conditions(La: float, Yb: float, Yw: float, sR: str = 'average') -> dict:
    """
    Calculates viewing condition parameters for CIECAM02.
    
    Args:
        La (float): Luminance of the adapting field (cd/m²).
        Yb (float): Luminance of the background (cd/m²).
        Yw (float): Luminance of the adopted white point (cd/m²).
        sR (str): Viewing surround condition ('average', 'dim', 'dark').
    
    Returns:
        dict: Computed parameters (FL, n, N_bb, N_cb).
    """
    # Constants for different surrounds (from Table 1 in the paper)
    surround_params = {
        'average': {'F': 1.0, 'c': 0.69, 'Nc': 1.0},
        'dim': {'F': 0.9, 'c': 0.59, 'Nc': 0.95},
        'dark': {'F': 0.8, 'c': 0.525, 'Nc': 0.8},
    }

    params = surround_params[sR]
    F = params['F']
    Nc = params['Nc']
    
    # Calculate FL (equations 1 and 2)
    k = 1 / (5 * La + 1)
    FL = 0.2 * (k**4) * (5*La) + 0.1 * ((1 - k**4)**2) * ((5 * La)**(1/3))
    
    # Calculate n, N_bb, N_cb (equations 3 and 4)
    n = Yb / Yw
    N_bb = N_cb = 0.725 * (1 / n) ** 0.2
    z = 1.48 + np.sqrt(n)
    
    return {
        'FL': FL, 'n': n, 'N_bb': N_bb, 'N_cb': N_cb, 'Nc': Nc, 'F': F, 'z': z}


def chromatic_adaptation(image_xyz: np.ndarray, white_point_xyz: np.ndarray, params: dict, La: float) -> np.ndarray:
    """
    Applies chromatic adaptation using the CAT02 transform.
    
    Args:
        image_xyz (np.ndarray): Input image in XYZ space.
        white_point_xyz (np.ndarray): White point in XYZ space.
        params (dict): Viewing condition parameters.
        La (float): Luminance of the adapting field (cd/m²).
    Returns:
        np.ndarray: Adapted image in LMS space.
    """
    # CAT02 matrix (equation 7)
    M_CAT02 = np.array([[0.7328, 0.4296, -0.1624],
                        [-0.7036, 1.6975, 0.0061],
                        [0.0030, 0.0136, 0.9834]])
    
    # Convert XYZ to LMS
    image_lms = np.dot(
        image_xyz.reshape(-1, 3), M_CAT02.T).reshape(image_xyz.shape)
    white_point_lms = np.dot(white_point_xyz, M_CAT02.T)
    
    # Apply D factor (degree of adaptation)
    D = params['F'] * (1 - (1/3.6) * np.exp((-La - 42)/92))
    adapted_lms = (D * (image_lms / white_point_lms) + (1 - D)) * image_lms.mean(axis=(0, 1))
    
    return adapted_lms

def nonlinear_compression(adapted_lms: np.ndarray, FL: float) -> np.ndarray:
    """
    Applies non-linear response compression to LMS values.
    
    Args:
        adapted_lms (np.ndarray): Adapted LMS values (H x W x 3).
        FL (float): Luminance adaptation factor.
    
    Returns:
        np.ndarray: Compressed LMS values.
    """
    # Constants based on the paper's formula (Eq. 13)
    FL_array = FL * adapted_lms
    
    # Apply non-linear compression (similar to hyperbolic function)
    compressed_lms = (400 * (FL_array / 100)**0.42) / (27.13 + (FL_array / 100)**0.42) + 0.1
    
    # Handle negative values
    compressed_lms[adapted_lms < 0] = -compressed_lms[adapted_lms < 0]
    return compressed_lms

def calculate_perceptual_attributes(compressed_lms: np.ndarray, params: dict) -> np.ndarray:
    """
    Computes lightness (J), chroma (C), and hue (h) from compressed LMS.
    
    Args:
        compressed_lms (np.ndarray): Compressed LMS values.
        params (dict): Viewing condition parameters.
    
    Returns:
        np.ndarray: Perceptual attributes (J, C, h).
    """
    # Calculate achromatic response A (Eq. 20)
    R, G, B = compressed_lms[..., 0], compressed_lms[..., 1], compressed_lms[..., 2]

    # Equation (14) + (15) 
    a = R - 12 * G / 11 + B / 11 
    b = (1/9) * (R + G -2*B) 

    # Hue angle h (Eq. 17)
    h = np.degrees(np.arctan(b / a)) 
    h = (h + 360) % 360  # Ensure hue is between 0 and 360

    # Equation (18)
    e = ((12500/13) * params['Nc'] * params['N_cb']) * (np.cos(h * (np.pi/180) + 2) + 3.8)

    # Equation (16) 
    t = (e * (a**2 + b**2)**(1/2)) / (R + G + (21/20)*B)

    # Equation (20)
    A = (2 * R + G + (1/20) * B - 0.305) * params['N_bb']
    
    # Lightness J (Eq. 21)
    J = 100 * (A / params['FL']) ** 0.5 # Should be (A/Aw)**cz
    
    # Chroma C (Eq. 23)
    # t = (np.abs(R - B) + np.abs(G - B)) / params['Nc']
    C = t**0.9 * np.sqrt(J / 100) * (1.64 - 0.29**params['n'])**0.73
    
    # # Hue angle h (Eq. 17)
    # h = np.degrees(np.arctan2(G, R))
    
    return np.stack((J, C, h), axis=-1)

# ========== Inverse CIECAM02 ========== #
def ciecam02_to_xyz(ciecam02_attr: np.ndarray, params: dict, white_point_lms: np.ndarray) -> np.ndarray:
    """
    Converts CIECAM02 attributes back to the XYZ color space.
    
    Args:
        ciecam02_attr (np.ndarray): CIECAM02 attributes (J, C, h).
        params (dict): Viewing condition parameters.
        white_point_lms (np.ndarray): White point in LMS space.
    
    Returns:
        np.ndarray: Reconstructed image in XYZ space.
    """
    J, C, h = ciecam02_attr[..., 0], ciecam02_attr[..., 1], ciecam02_attr[..., 2]
    
    # Reconstruct LMS values
    L = (J / 100) ** 2 * params['FL']**0.25 # Eq.(22)
    M = C * np.sin(np.radians(h))
    S = C * np.cos(np.radians(h))
    
    reconstructed_lms = np.stack((L, M, S), axis=-1)
    
    # Inverse CAT02 transform (Eq. 7)
    M_CAT02_inv = np.linalg.inv(
        np.array([
            [0.7328, 0.4296, -0.1624],
            [-0.7036, 1.6975, 0.0061],
            [0.0030, 0.0136, 0.9834]]))
    image_xyz = np.dot(reconstructed_lms.reshape(-1, 3), M_CAT02_inv.T).reshape(ciecam02_attr.shape)
    return np.clip(image_xyz, 0, 1)
