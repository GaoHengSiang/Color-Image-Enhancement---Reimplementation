#generate results
import cv2 
import numpy as np
from PIL import Image
from ciecam02 import CIECAM02 
import matplotlib.pyplot as plt
from parameters import constants, matrices, colordata
from low_backlight import *
from nightlight_sim import *
from pathlib import Path

def low_backlight(src: np.ndarray, do_map: bool):
    # Load the image
    image_bgr = src

    # Convert the image from BGR to RGB if it exists. 
    if image_bgr is not None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  
    else:
        raise FileNotFoundError("Image does not exist.") 
    shape = image_rgb.shape 

    model = CIECAM02(constants, matrices, colordata) # default configuration

    # Forward pass 
    print("Convert RGB_i to XYZ_i...")
    RGB_i = image_rgb/255.0
    XYZ_i = device_rgb_to_xyz(RGB_i, M_f, gamma_rf, gamma_gf, gamma_bf) 

    # Configure the model according to the surrounding condition
    print("Forward pass through the color model...")
    model.configure(white="full_light", surround="average", light="high", bg="high")
    XYZ_i = XYZ_i.reshape(-1, 3)
    JCh = model.xyz_to_ciecam02(XYZ_i) 

    # Reverse back to a new image
    print("Reverse pass through the color model...")
    model.configure(white="low_light", surround="average", light="high", bg="high")  
    XYZ_e = model.inverse_model(JCh).reshape(shape)

    #post gamut mapping
    if do_map:
        print("post gamut mapping...")
        JCh_reshape = JCh.reshape(shape)
        JC = JCh_reshape[..., 0] * JCh_reshape[..., 1]
        # Normalize array to [0, 1] range
        normalized_JC = (JC - np.min(JC)) / (np.max(JC) - np.min(JC))
        RGB_prime = post_gamut_mapping(RGB_i, XYZ_e, normalized_JC)
    else:
        print("\033[33m[WARNING]\033[0m post gamut mapping disabled, use -m, --map to enable")
        RGB_prime = device_xyz_to_rgb(XYZ_e, M_l, gamma_rl, gamma_gl, gamma_bl)
    # rgb = jch2rgb(JCH).reshape(shape)
    print("Simulate low backlight for original image...")
    without_enhancement = simulated_low_backlight(RGB_i, M_l, gamma_rl, gamma_gl, gamma_bl)
    print("Simulate low backlight for enhanced image...")
    with_enhancement = simulated_low_backlight(RGB_prime, M_l, gamma_rl, gamma_gl, gamma_bl)
    
    without_enhancement = np.round(without_enhancement*255).astype(np.uint8)
    without_enhancement = cv2.cvtColor(without_enhancement, cv2.COLOR_RGB2BGR)
    with_enhancement = np.round(with_enhancement*255).astype(np.uint8)
    with_enhancement = cv2.cvtColor(with_enhancement, cv2.COLOR_RGB2BGR)

    return without_enhancement, with_enhancement

def nightlight(src: np.ndarray,strength: int, do_map: bool, do_compensation: bool):
    """
    manual: 
        python ./nightlight_compen_demo.py -h
    example:
        python ./nightlight_compen_demo.py -i ./Lenna.png -s 100 -m -c
    """
    # Load the image
    image_bgr = src

    # Convert the image from BGR to RGB if it exists. 
    if image_bgr is not None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  
    else:
        raise FileNotFoundError("Image does not exist.") 
    shape = image_rgb.shape 

    model = CIECAM02(constants, matrices, colordata) # default configuration
    
    #dynamically set white point
    temperature = map_windows_nightlight_slider_to_temp(strength)
    channel_gain = kelvin_to_scaling_factors(temperature)
    channel_gain = np.array(channel_gain).reshape(1, 1, 3)
    nl_whitepoint = device_rgb_to_xyz(channel_gain, M_f, gamma_rf, gamma_gf, gamma_bf)
    nl_whitepoint = list(nl_whitepoint.reshape(3))
    constants["whitepoint"]["night_light"] = nl_whitepoint
    fl_white_point = constants["whitepoint"]["full_light"]
    print(f"full light white point: {fl_white_point}")
    print(f"night light white point: {nl_whitepoint}")

    # Forward pass 
    print("Convert RGB_i to XYZ_i...")
    RGB_i = image_rgb/255.0
    XYZ_i = device_rgb_to_xyz(RGB_i, M_f, gamma_rf, gamma_gf, gamma_bf) 

    # Configure the model according to the surrounding condition
    print("Forward pass through the color model...")
    model.configure(white="full_light", surround="average", light="high", bg="high")
    XYZ_i = XYZ_i.reshape(-1, 3)
    JCh = model.xyz_to_ciecam02(XYZ_i) 

    # Reverse back to a new image
    print("Reverse pass through the color model...")
    model.configure(white="night_light", surround="average", light="high", bg="high")  
    XYZ_e = model.inverse_model(JCh).reshape(shape)

    #Inverse Device Full Light (enhanced xyz back to rgb)
    print("enhanced XYZ to RGB...")
    RGB_nl = device_xyz_to_rgb(XYZ_e, M_f, gamma_rf, gamma_gf, gamma_bf)

    #Inverse Channel Gain
    print("Inverse Channel Gain, handling zero denominator, clipping...")
    safe_gain = np.clip(channel_gain, 1e-5, 1)
    RGB_nl = RGB_nl / safe_gain
    if do_compensation:
        print("Applying inverse gain twice to counteract nightlight, effects may vary")
        RGB_nl = RGB_nl / safe_gain
    else:
        print("\033[33m[WARNING]\033[0m inverse gain compensation disabled, use -c, --compensate to enable")
    RGB_clip = np.clip(RGB_nl, 0, 1)
    #RGB_clip = RGB_nl

    #Post Gamut Mapping
    if do_map:
        print("Post Gamut Mapping enabled")
        JCh_reshape = JCh.reshape(shape)
        JC = JCh_reshape[..., 0] * JCh_reshape[..., 1]
        # Normalize JC to [0, 1] range
        normalized_JC = (JC - np.min(JC)) / (np.max(JC) - np.min(JC))
        normalized_JC = np.expand_dims(normalized_JC, axis=-1)  # Shape becomes (512, 512, 1)
        RGB_prime = RGB_i*normalized_JC + RGB_clip*(1-normalized_JC)
    else:
        print("\033[33m[WARNING]\033[0m post Gamut Mapping disabled, use -m, --map to enable...")
        RGB_prime = RGB_clip

    #print(np.max(RGB_prime))
    
    
    print("Simulate nightlight for original image...")
    without_enhancement = temp_by_channel_gain_cont(RGB_i, temperature)
    print("Simulate nightlight for enhanced image...")
    with_enhancement = temp_by_channel_gain_cont(RGB_prime, temperature)
    
    without_enhancement = np.round(without_enhancement*255).astype(np.uint8)
    without_enhancement = cv2.cvtColor(without_enhancement, cv2.COLOR_RGB2BGR)
    with_enhancement = np.round(with_enhancement*255).astype(np.uint8)
    with_enhancement = cv2.cvtColor(with_enhancement, cv2.COLOR_RGB2BGR)
    
    return without_enhancement, with_enhancement
    
    
if __name__ == "__main__":
    image_folder = "./test_images/"
    color_images = ["baboon.png", "fruits.png", "HappyFish.jpg", "Lenna.png", "nave.jpg",
                    "parrot.jfif", "parrot2.jpg", "peppers.png", "rosette.jpg", "tulips.png"]
    destination_folder = "./results/"
    
    do_map = [False, True]
    do_compen = [False, True]
    nightlight_str = [25, 50, 75, 100]

    """
    #generate low light images
    for _map in do_map:
        for _image in color_images:
            image = cv2.imread(image_folder+_image)
            vanilla, enhanced = low_backlight(image, _map)
            
            if _map:
                #write to map
                cv2.imwrite(destination_folder+"lowbacklight/map/"+_image+".png", enhanced)
                cv2.imwrite(destination_folder+"lowbacklight/vanilla/"+_image+".png", vanilla)
            else:
                #write to no_map
                cv2.imwrite(destination_folder+"lowbacklight/no_map/"+_image+".png", enhanced)
                cv2.imwrite(destination_folder+"lowbacklight/vanilla/"+_image+".png", vanilla)
    """
    #generate nightlight images  --> 2*2*4*10 = 160 images
    for _map in do_map:
        for _do_comp in do_compen:
            for _str in nightlight_str:
                for _image in color_images:
                    image = cv2.imread(image_folder+_image)
                    vanilla, enhanced = nightlight(src=image, strength=_str, do_map=_map, do_compensation=_do_comp)

                    folder_path = Path(destination_folder+f"nightlight/{_str}/vanilla/")
                    # Create the folder if it doesn't exist
                    folder_path.mkdir(parents=True, exist_ok=True)

                    cv2.imwrite(destination_folder+f"nightlight/{_str}/vanilla/"+_image+".png", vanilla)
                    
                    output_path = destination_folder+"nightlight/"+f"{_str}/"
                    if _map:
                        output_path = output_path+"map/"
                    else:
                        output_path = output_path+"no_map/"

                    if _do_comp:
                        output_path = output_path+"comp/"
                    else:
                        output_path = output_path+"no_comp/"
                    
                    folder_path = Path(output_path)
                    # Create the folder if it doesn't exist
                    folder_path.mkdir(parents=True, exist_ok=True)

                    cv2.imwrite(output_path+_image+".png", enhanced)
                    

