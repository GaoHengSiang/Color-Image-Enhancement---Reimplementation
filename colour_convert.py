import cv2
import numpy as np 
from PIL import Image 

def xyz2infinitergb(xyz):
    xyz = xyz/100.0

    M_1 = np.array([[3.2406, -1.5372, -0.4986],
                    [-0.9689, 1.8758,  0.0415],
                    [0.0557, -0.2040,  1.0570]]).T
    RGB = xyz.dot(M_1)
    RGB = np.where(RGB <= 0, 0.00000001, RGB)
    RGB = np.where(
        RGB > 0.0031308,
        1.055*(RGB**0.4166666)-0.055,
        12.92*RGB)

    RGB = np.around(RGB*255)

    return RGB

def xyz2rgb(xyz):
    xyz = xyz/100.0

    M_1 = np.array([[3.2406, -1.5372, -0.4986],
                    [-0.9689, 1.8758,  0.0415],
                    [0.0557, -0.2040,  1.0570]]).T
    RGB = xyz.dot(M_1)
    RGB = np.where(RGB <= 0, 0.00000001, RGB)
    RGB = np.where(
        RGB > 0.0031308,
        1.055*(RGB**0.4166666)-0.055,
        12.92*RGB)

    RGB = np.around(RGB*255)
    RGB = np.where(RGB <= 0, 0, RGB)
    RGB = np.where(RGB > 255, 255, RGB)
    RGB = RGB.astype('uint8')

    return RGB

def rgb2xyz(color):
    color = color/255.0
    color = np.where(
        color > 0.04045,
        np.power(((color+0.055)/1.055), 2.4),
        color/12.92)
    M = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]]).T

    return color.dot(M)*100


# ========== Constant Parameters ========== #
whitepoint = {
    'white': [95.05, 100.00, 108.88],
    'c': [109.85, 100.0, 35.58]}

surround_params = {
    'average': {'F': 1.0, 'c': 0.69, 'Nc': 1.0},
    'dim': {'F': 0.9, 'c': 0.59, 'Nc': 0.95},
    'dark': {'F': 0.8, 'c': 0.525, 'Nc': 0.8},
}

light_intensity = {'default': 80.0, 'high': 318.31, 'low': 31.83}
bg_intensity = {'default': 16.0, 'high': 20.0, 'low': 10.0}

current_white = whitepoint['white']  # list
current_env = surround_params['average']  # dict
current_light = light_intensity['default']  # float
current_bg = bg_intensity['default']  # float

# ========== Configuration ========== #
def setconfig(a='white', b='average', c='default', d='default'):
    current_white = whitepoint[a]
    current_env = surround_params[b]
    current_light = light_intensity[c]
    current_bg = bg_intensity[d]

# ========== Step 0: Calculate Constant Parameters =========== #
Xwr, Ywr, Zwr = 100, 100, 100 # reference white in reference illuminant
Mcat02 = np.array([
    [0.7328, 0.4296, -0.1624],
    [-0.7036, 1.6975, 0.0061],
    [0.0030, 0.0136, 0.9834]])
Xw, Yw, Zw = current_white # Adopted white in test illuminant
Nc, c, F = current_env["Nc"], current_env["c"], current_env["F"]
LA = current_light # luminance of test-adapting field 
Yb = current_bg # background in test conditions

# Independent parameters 
Rw, Gw, Bw = Mcat02.dot(np.array([Xw, Yw, Zw]))
D = F * (1 - (1/3.6) * (np.exp((-LA - 42)/92)))

# Constraint for D 
if D > 1:
    D = 1
if D < 0:
    D = 0

Dr, Dg, Db = [
    Yw * D/Rw + (1-D), 
    Yw * D/Gw + (1-D), 
    Yw * D/Bw + (1-D)]
Rwc, Gwc, Bwc = [Dr*Rw, Dg*Gw, Db*Bw]
k = 1/  (5*LA + 1)
FL = 0.2*(k**4)*(5*LA) + 0.1*((1-(k**4))**2)*((5*LA)**(1/3.0)) 

n = Yb/Yw
if n > 1:
    n = 1
if n < 0:
    n = 0.000001

Nbb = Ncb = 0.725*((1.0/n)**0.2)
z = 1.48 + n**0.5
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
Rw_, Gw_, Bw_ = Mhpe.dot(inv_Mcat02.dot([Rwc, Gwc, Bwc]))

# Unique hue data for calculation of hue quadrature (Table 2.4)
colordata = [
    [20.14, 0.8, 0], 
    [90, 0.7, 100], 
    [164.25, 1.0, 200],
    [237.53, 1.2, 300],
    [380.14, 0.8, 400]]

Rwa_ = (400 * ((FL*Rw_/100)**0.42))/(27.13+((FL*Rw_/100)**0.42))+0.1
Gwa_ = (400 * ((FL*Gw_/100)**0.42))/(27.13+((FL*Gw_/100)**0.42))+0.1
Bwa_ = (400 * ((FL*Bw_/100)**0.42))/(27.13+((FL*Bw_/100)**0.42))+0.1
Aw = Nbb * (2*Rwa_+Gwa_+(Bwa_/20) - 0.305)

# ========== XYZ to CIECAM02 (Forward) ========== #
def xyz2cam02(XYZ):
    # step 1: calculate cone responses
    RGB = XYZ.dot(Mcat02.T)

    # step 2: calculate the corresponding cone responses
    RcGcBc = RGB * np.array([Dr, Dg, Db])

    # step 3: calculate Hunt-Pointer-Estevex response
    R_G_B_ = RcGcBc.dot(inv_Mcat02.T).dot(Mhpe.T)

    # step 4: calculate the post-adaptation cone response
    R_G_B_in = np.power(FL*R_G_B_/100, 0.42)
    Ra_Ga_Ba_ = (400 * R_G_B_in)/(27.13 + R_G_B_in) + 0.1

    # step 5: calculate redness-greeness (a), yellowness-blueness (b) components and hue angle (h)
    a = Ra_Ga_Ba_[:, 0] - 12*Ra_Ga_Ba_[:, 1]/11 + Ra_Ga_Ba_[:, 2]/11
    b = (1/9.0) * (Ra_Ga_Ba_[:, 0] + Ra_Ga_Ba_[:, 1] - 2*Ra_Ga_Ba_[:, 2])
    h = np.arctan2(b, a) # 4-quadrant inverse, range [-pi, pi]
    # Convert to degree in range [0, 360]
    h = np.where(h < 0, (h+np.pi*2)*180/np.pi, h*180/np.pi)

    # step 6: calculate eccentricity (etemp) and hue composition (H)
    h_prime = np.where(h < colordata[0][0], h+360, h)
    etemp = (np.cos(h_prime*np.pi/180 + 2) + 3.8) * (1/4)
    # List of values for h_prime
    coarray = np.array([20.14, 90, 164.25, 237.53, 380.14])
    position_ = coarray.searchsorted(h_prime)

    def TransferHue(h_prime, i):
        data_i = colordata[i-1]
        h_i, e_i, H_i = data_i[0], data_i[1], data_i[2]
        data_i1 = colordata[i]
        h_i1, e_i1 = data_i1[0], data_i1[1]
        Hue = H_i + (
            (100 * (h_prime-h_i)/e_i) /
            (((h_prime-h_i)/e_i) + (h_i1-h_prime)/e_i1))
        return Hue
    
    ufunc_TransferHue = np.frompyfunc(TransferHue, 2, 1)
    H = ufunc_TransferHue(h_prime, position_).astype('float')

    # step 7: calcualte achromatic response (A) 
    A = Nbb * (
        2*Ra_Ga_Ba_[:, 0] + Ra_Ga_Ba_[:, 1] + (Ra_Ga_Ba_[:, 2]/20.0) - 0.305)
    
    # step 8: calcualte the correlate of lightness (J)
    J = 100 * ((A/Aw)**(c*z))

    # step 9: calculate the correlate of brightness (Q)
    Q = (4/c) * ((J/100.0)**0.5) * (Aw + 4) * (FL**0.25)

    # step 10: calcualte the correlates of chroma (C), colourfulness (M), and saturation (s)
    t = ((50000/13.0)*Nc*Ncb*etemp*((a**2+b**2)**0.5)) /\
        (Ra_Ga_Ba_[:, 0]+Ra_Ga_Ba_[:, 1]+(21/20.0)*Ra_Ga_Ba_[:, 2])
    C = t**0.9*((J/100.0)**0.5)*((1.64-(0.29**n))**0.73)
    M = C*(FL**0.25)
    s = 100*((M/Q)**0.5)
    return np.array([h, H, J, Q, C, M, s]).T

def rgb2jch(color):
    XYZ = rgb2xyz(color)
    value = xyz2cam02(XYZ)
    return value[:, [2, 4, 1]]*np.array([1.0, 1.0, 0.9])

# ========== Inverse Model ========== #
def jch2xyz(JCH):
    # step 1: obtain J, C, and h (J and C are available)
    JCH = JCH*np.array([1.0, 1.0, 10/9.0])
    J = JCH[:, 0]
    C = JCH[:, 1]
    H = JCH[:, 2]
    coarray = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    position_ = coarray.searchsorted(H)
    # Calculate h from H 
    def TransferHue(H_, i):
        C1 = colordata[i-1]
        C2 = colordata[i]
        h = ((H_-C1[2])*(C2[1]*C1[0]-C1[1]*C2[0])-100*C1[0]*C2[1]) /\
            ((H_-C1[2])*(C2[1]-C1[1]) - 100*colordata[i][1])
        if h > 360:
            h -= 360
        return h
    
    ufunc_TransferHue = np.frompyfunc(TransferHue, 2, 1)
    h_deg = ufunc_TransferHue(JCH[:, 2], position_).astype('float')
    J = np.where(J <= 0, 0.00001, J)
    C = np.where(C <= 0, 0.00001, C)
    
    # step 2: calculate t, etemp, p1, p2, and p3
    t = (C/(((J/100.0)**0.5)*((1.64-(0.29**n))**0.73)))**(1/0.9)
    t = np.where(t - 0 < 0.00001, 0.00001, t)
    etemp = (np.cos(h_deg*np.pi/180 + 2) + 3.8) * (1/4)
    A = Aw*((J/100)**(1/(c*z)))
    p1 = ((50000/13.0) * Nc * Ncb) * etemp * (1/t)
    p2 = A/Nbb + 0.305
    p3 = 21/20.0
    h_rad = h_deg*np.pi/180

    # step 3: calculate a and b
    def evalAB(h, p1, p2):
        # TODO: Add condition t=0
        if abs(np.sin(h)) >= abs(np.cos(h)):
            p4 = p1/np.sin(h)
            b = (p2*(2+p3)*(460.0/1403)) / (
                    p4+(2+p3)*(220.0/1403)*(np.cos(h)/np.sin(h)) - 27.0/1403 +
                    p3*(6300.0/1403))
            a = b*(np.cos(h)/np.sin(h))
        else:  # abs(np.cos(h))>abs(np.sin(h)):
            p5 = p1/np.cos(h)
            a = (p2*(2+p3)*(460.0/1403)) / (
                    p5+(2+p3)*(220.0/1403) - (
                        27.0/1403 - p3*(6300.0/1403))*(np.sin(h)/np.cos(h)))
            b = a*(np.sin(h)/np.cos(h))
        return np.array([a, b])
    
    ufunc_evalAB = np.frompyfunc(evalAB, 3, 1)
    abinter = np.vstack(ufunc_evalAB(h_rad, p1, p2))
    a = abinter[:, 0]
    b = abinter[:, 1]
    
    # step 4: calculate Ra_, Ga_, and Ba_
    Ra_ = (460*p2 + 451*a + 288*b)/1403.0
    Ga_ = (460*p2 - 891*a - 261*b)/1403.0
    Ba_ = (460*p2 - 220*a - 6300*b)/1403.0

    # step 5: calculate R_, G_, and B_
    R_ = np.sign(Ra_-0.1)*(100.0/FL) * (
        ((27.13*np.abs(Ra_-0.1))/(400-np.abs(Ra_-0.1)))**(1/0.42))
    G_ = np.sign(Ga_-0.1)*(100.0/FL) * (
        ((27.13*np.abs(Ga_-0.1))/(400-np.abs(Ga_-0.1)))**(1/0.42))
    B_ = np.sign(Ba_-0.1)*(100.0/FL) * (
        ((27.13*np.abs(Ba_-0.1))/(400-np.abs(Ba_-0.1)))**(1/0.42))
    
    # step 6: calculate Rc, Gc, and Bc
    RcGcBc = (np.array([R_, G_, B_]).T).dot(inv_Mhpe.T).dot(Mcat02.T)

    # step 7: calculate R, G, and B 
    RGB = RcGcBc/np.array([Dr, Dg, Db])

    # step 8: calculate X, Y, and Z 
    XYZ = RGB.dot(inv_Mcat02.T)

    return XYZ

def jch2rgb(jch):
    xyz = jch2xyz(jch)
    return xyz2rgb(xyz)


# ========== Example Run ========== #
if __name__ == "__main__":

    # forward pass 
    # im = Image.open("./Lenna.png")
    # rgb = np.array(im)
    
    # Load the image
    image_bgr = cv2.imread("./Lenna.png")

    # Convert the image from BGR to RGB if it exists. 
    if image_bgr is not None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  
    else:
        raise FileNotFoundError("Image does not exist.") 
    shape = image_rgb.shape 
    
    JCH = rgb2jch(image_rgb.reshape(-1, 3))

    # reverse back to a new image
    rgb = jch2rgb(JCH).reshape(shape)
    im = Image.fromarray(rgb)
    im.show()