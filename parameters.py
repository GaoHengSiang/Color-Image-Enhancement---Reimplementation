import numpy as np 

# ========== Constant Parameters ========== #
whitepoint = {
    'white': [95.05, 100.00, 108.88],
    'c': [109.85, 100.0, 35.58]}

surround_params = {
    'average': {'F': 1.0, 'c': 0.69, 'Nc': 1.0},
    'dim': {'F': 0.9, 'c': 0.59, 'Nc': 0.95},
    'dark': {'F': 0.8, 'c': 0.535, 'Nc': 0.8},
}

light_intensity = {'default': 80.0, 'high': 318.31, 'low': 31.83}
bg_intensity = {'default': 16.0, 'high': 20.0, 'low': 10.0}
# Reference white in reference illuminant
Xwr, Ywr, Zwr = 100, 100, 100 

Mcat02 = np.array([
    [0.7328, 0.4296, -0.1624],
    [-0.7036, 1.6975, 0.0061],
    [0.0030, 0.0136, 0.9834]])
inv_Mcat02 = np.array([
    [1.096241, -0.278869, 0.182745],
    [0.454369, 0.473533, 0.072098],
    [-0.009628, -0.005698, 1.015326]])
Mhpe = np.array([
    [0.38971, 0.68898, -0.07868],
    [-0.22981, 1.18340, 0.04641],
    [0.00000, 0.00000, 1.00000]])
inv_Mhpe = np.array([
    [1.910197, -1.112124, 0.201908],
    [0.370950, 0.629054, -0.000008],
    [0.000000, 0.000000, 1.000000]])
# Unique hue data for calculation of hue quadrature (Table 2.4)
colordata = [
    [20.14, 0.8, 0], 
    [90, 0.7, 100], 
    [164.25, 1.0, 200],
    [237.53, 1.2, 300],
    [380.14, 0.8, 400]]

# ===== Grouping for Convenience ===== #
constants = {
    "whitepoint": whitepoint, 
    "surround": surround_params,
    "light_intensity": light_intensity,
    "bg_intensity": bg_intensity, 
    "reference_white": (Xwr, Ywr, Zwr)
}

matrices = {
    "CAT02": Mcat02,
    "inv_CAT02": inv_Mcat02, 
    "HPE": Mhpe, 
    "inv_HPE": inv_Mhpe
}