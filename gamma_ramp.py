import ctypes
from ctypes import wintypes

def get_gamma_ramp():
    hdc = ctypes.windll.user32.GetDC(0)  # Get the device context of the screen
    gamma_array = (wintypes.WORD * 256)()
    if ctypes.windll.gdi32.GetDeviceGammaRamp(hdc, gamma_array):
        return gamma_array[:]
    return None

gamma_ramp = get_gamma_ramp()
if gamma_ramp:
    print("Gamma Ramp Data:", gamma_ramp)
else:
    print("Unable to retrieve gamma ramp data.")
