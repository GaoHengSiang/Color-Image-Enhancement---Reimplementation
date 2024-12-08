import numpy as np 

class CIECAM02:
    """
    Implementation of the forward CIECAM02 color appearance model.
    """

    def __init__(self, constants: dict, matrices: dict, colordata: list):
        """
        Initializes the CIECAM02 model with shared constants and matrices.

        Args:
            constants (dict): Dictionary containing configuration parameters 
                such as white points, surround parameters, light intensity, and background intensity.
            matrices (dict): Dictionary containing precomputed matrices 
                (Mcat02, inv_Mcat02, Mhpe, inv_Mhpe, etc.).
            color_data (list): Unique hue data for calculation of hue quadrature
        """
        self.constants = constants
        self.colordata = colordata
        
        self.Mcat02 = matrices["CAT02"]
        self.inv_Mcat02 = matrices["inv_CAT02"] 
        self.Mhpe = matrices["HPE"]
        self.inv_Mhpe = matrices["inv_HPE"]

        # Initialize default configurations
        self.current_white = self.constants["whitepoint"]["white"]
        self.current_env = self.constants["surround"]["average"]
        self.current_light = self.constants["light_intensity"]["default"]
        self.current_bg = self.constants["bg_intensity"]["default"]

    def configure(self, white="white", surround="average", light="default", bg="default"):
        """
        Configures the model with specific environmental parameters.

        Args:
            white (str): Key for the white point to use.
            surround (str): Key for the surround parameters to use.
            light (str): Key for the luminance of the adapting field.
            bg (str): Key for the background intensity.
        """
        self.current_white = self.constants["whitepoint"][white]
        self.current_env = self.constants["surround"][surround]
        self.current_light = self.constants["light_intensity"][light]
        self.current_bg = self.constants["bg_intensity"][bg]

    def calculate_independent_parameters(self):
        """
        Calculates input-independent parameters needed for CIECAM02 transformations.

        Returns:
            dict: A dictionary of independent parameters.
        """
        Xw, Yw, Zw = self.current_white
        Nc, c, F = self.current_env["Nc"], self.current_env["c"], self.current_env["F"]
        LA = self.current_light
        Yb = self.current_bg

        # Compute chromatic adaptation parameters
        Rw, Gw, Bw = self.Mcat02.dot([Xw, Yw, Zw])
        D = F * (1 - (1 / 3.6) * np.exp((-LA - 42) / 92))
        D = np.clip(D, 0, 1)

        Dr, Dg, Db = Yw * D / Rw + (1 - D), Yw * D / Gw + (1 - D), Yw * D / Bw + (1 - D)
        Rwc, Gwc, Bwc = Dr * Rw, Dg * Gw, Db * Bw

        # Compute the viewing condition parameters
        k = 1 / (5 * LA + 1)
        FL = 0.2 * (k**4) * (5 * LA) + 0.1 * ((1 - k**4)**2) * ((5 * LA)**(1 / 3.0))

        n = Yb / Yw
        n = np.clip(n, 0.000001, 1)
        Nbb = Ncb = 0.725 * (1 / n)**0.2
        z = 1.48 + n**0.5

        Rw_, Gw_, Bw_ = self.Mhpe.dot(self.inv_Mcat02.dot([Rwc, Gwc, Bwc]))

        # TODO: Refactor the following three lines into a function
        Rwa_ = (400 * (FL * Rw_ / 100)**0.42) / (27.13 + (FL * Rw_ / 100)**0.42) + 0.1
        Gwa_ = (400 * (FL * Gw_ / 100)**0.42) / (27.13 + (FL * Gw_ / 100)**0.42) + 0.1
        Bwa_ = (400 * (FL * Bw_ / 100)**0.42) / (27.13 + (FL * Bw_ / 100)**0.42) + 0.1

        Aw = Nbb * (2 * Rwa_ + Gwa_ + Bwa_ / 20 - 0.305)

        return {
            "D": D, "Dr": Dr, "Dg": Dg, "Db": Db,
            "FL": FL, "n": n, "Nbb": Nbb, "Ncb": Ncb, 
            "Nc": Nc, "F": F, "z": z, "Aw": Aw, "c": c
        }
    
    def transfer_hue(self, h_prime, i):
        data_i = self.colordata[i-1]
        h_i, e_i, H_i = data_i[0], data_i[1], data_i[2]
        data_i1 = self.colordata[i]
        h_i1, e_i1 = data_i1[0], data_i1[1]
        Hue = H_i + (
            (100 * (h_prime-h_i)/e_i) /
            (((h_prime-h_i)/e_i) + (h_i1-h_prime)/e_i1))
        return Hue
    
    def xyz_to_ciecam02(self, XYZ):
        """
        Converts XYZ tristimulus values to CIECAM02 model.

        Args:
            XYZ (numpy.ndarray): Array of XYZ values.

        Returns:
            numpy.ndarray: Array of CIECAM02 attributes.
        """
        params = self.calculate_independent_parameters()
        # Dr, Dg, Db = params["Dr"], params["Dg"], params["Db"]
        FL, Aw, z = params["FL"], params["Aw"], params["z"]
        Nbb, Ncb, n = params["Nbb"], params["Ncb"], params["n"]

        # Step 1: Chromatic adaptation
        RGB = XYZ.dot(self.Mcat02.T)
        # Step 2: Calculate the corresponding cone responses
        RcGcBc = RGB * [params["Dr"], params["Dg"], params["Db"]]
        # Step 3: Calculate Hunt-Pointer-Estevex response
        R_G_B_ = RcGcBc.dot(self.inv_Mcat02.T).dot(self.Mhpe.T)

        # Step 4: Post-adaptation response
#EDIT: handle negative R_G_B_ values
        R_G_B_in = np.power(FL * np.abs(R_G_B_) / 100, 0.42)
                            
        Ra_Ga_Ba_ = np.where(R_G_B_ >= 0,
                            (400 * R_G_B_in) / (27.13 + R_G_B_in) + 0.1,
                            (-400 * R_G_B_in) / (27.13 + R_G_B_in) + 0.1)

        # Step 5: Calculate redness-greeness (a), yellowness-blueness (b) components and hue angle (h)
        a = Ra_Ga_Ba_[:, 0] - 12 * Ra_Ga_Ba_[:, 1] / 11 + Ra_Ga_Ba_[:, 2] / 11
        b = (Ra_Ga_Ba_[:, 0] + Ra_Ga_Ba_[:, 1] - 2 * Ra_Ga_Ba_[:, 2]) / 9
        h = np.arctan2(b, a) * (180 / np.pi)
        h = np.where(h < 0, h + 360, h)

        # Step 6: Calculate eccentricity (etemp) and hue composition (H)
        h_prime = np.where(h < self.colordata[0][0], h+360, h)
        etemp = (np.cos(h_prime * np.pi/180 + 2) + 3.8) * (1/4)
        # List of values for h_prime
        coarray = np.array([20.14, 90, 164.25, 237.53, 380.14])
        position_ = coarray.searchsorted(h_prime)
        ufunc_TransferHue = np.frompyfunc(self.transfer_hue, 2, 1)
        H = ufunc_TransferHue(h_prime, position_).astype('float')   

        # Step 7: Calcualte achromatic response (A) 
        A = params["Nbb"] * (
            2*Ra_Ga_Ba_[:, 0] + Ra_Ga_Ba_[:, 1] + Ra_Ga_Ba_[:, 2] / 20 - 0.305)
        
        # Step 8: Calcualte the correlate of lightness (J)
        J = 100 * (A / Aw)**(params["c"]*params["z"])

        # Step 9: Calculate the correlate of brightness (Q)
        Q = (4 / params["c"]) * ((J / 100)**0.5) * (Aw + 4)*  (FL**0.25)

        # Step 10 (Optional): Calcualte the correlates of chroma (C), colourfulness (M), and saturation (s)
        t = (
            (50000/13.0)*params["Nc"]*params["Ncb"]*etemp*((a**2+b**2)**0.5)) /\
            (Ra_Ga_Ba_[:, 0]+Ra_Ga_Ba_[:, 1]+(21/20.0)*Ra_Ga_Ba_[:, 2])
        C = t**0.9*((J/100.0)**0.5)*((1.64-(0.29**params["n"]))**0.73)
        M = C*(FL**0.25)
        s = 100*((M/Q)**0.5)

        # We only need to return 3 out of 7 components calculated. Here, I chose J, Q, H. The inverse will be built upon these three components. 
# Teacher's slides uses JCh, will that make a difference?
# might be a typo here: Q --> C
#why times 1.0, 1.0, 0.9?
        return np.array([J, Q, H]).T*np.array([1.0, 1.0, 0.9]) # np.array([h, H, J, Q, C, M, s]).T
    
    def inverse_transfer_hue(self, H_, coarray):
        position = coarray.searchsorted(H_)
        C1 = self.colordata[position - 1]
        C2 = self.colordata[position]
        h = ((H_-C1[2])*(C2[1]*C1[0]-C1[1]*C2[0])-100*C1[0]*C2[1]) /\
            ((H_-C1[2])*(C2[1]-C1[1]) - 100*C2[1])
        if h > 360:
            h -= 360
        return h
    
    def inverse_model(self, JCH):
        """
        Converts JCH (Lightness, Chroma, Hue) to XYZ color space using the CIECAM02 model.

        Args:
            JCH (numpy.ndarray): Array of shape (N, 3) containing J (Lightness), 
                                C (Chroma), and H (Hue in CAM02).

        Returns:
            numpy.ndarray: Array of shape (N, 3) containing XYZ tristimulus values.
        """
        # Step 1: Extract J, C, and H and handle scaling for input format
        JCH = JCH * np.array([1.0, 1.0, 10 / 9.0])
        J, C, H = JCH[:, 0], JCH[:, 1], JCH[:, 2]
        # Clip J and C to avoid numerical issues
        J = np.maximum(J, 1e-5)
        C = np.maximum(C, 1e-5)

        coarray = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
#is it possible to just pass all attributes (only relative ones are allowed?)

        # position_ = coarray.searchsorted(H)
        # ufunc_TransferHue = np.frompyfunc(self.inverse_transfer_hue, 2, 1)
        # h_deg = ufunc_TransferHue(JCH[:, 2], position_).astype('float')
        h_deg = np.array([self.inverse_transfer_hue(H_i, coarray) for H_i in H])

        # Step 2: Calculate t, A, and p1, p2, p3 based on J, C, and H
        params = self.calculate_independent_parameters()
        t = (C / ((J / 100.0) ** 0.5 * ((1.64 - 0.29 ** params["n"]) ** 0.73))) ** (1 / 0.9)
        t = np.maximum(t, 1e-5)
# should we handle t = 0 by setting p1 to 0 ?
        etemp = (np.cos(h_deg*np.pi/180 + 2) + 3.8) * (1 / 4)
        A = params["Aw"] * (J / 100) ** (1 / (params["c"] * params["z"]))
        p1 = ((50000 / 13.0) * params["Nc"] * params["Ncb"] * etemp) / t
        p2 = A / params["Nbb"] + 0.305
        p3 = 21 / 20.0
        h_rad = np.radians(h_deg)

# What is the scope of this function?
#should pass p3 as argument, will it alter the behavior of frompyfunc?
        # Step 3: Compute a and b
        def compute_a_b(t, h, p1, p2):
            if t == 0:
                a, b = 0, 0
            elif np.abs(np.sin(h)) >= np.abs(np.cos(h)):
                p4 = p1 / np.sin(h)
                b = (p2 * (2 + p3) * (460.0 / 1403)) / (
                    p4 + (2 + p3) * (220.0 / 1403) * (np.cos(h) / np.sin(h)) - 27.0 / 1403 + p3 * (6300.0 / 1403)
                )
                a = b * (np.cos(h) / np.sin(h))
            else:
                p5 = p1 / np.cos(h)
                a = (p2 * (2 + p3) * (460.0 / 1403)) / (
                    p5 + (2 + p3) * (220.0 / 1403) - (27.0 / 1403 - p3 * (6300.0 / 1403)) * (np.sin(h) / np.cos(h))
                )
                b = a * (np.sin(h) / np.cos(h))
            return np.array([a, b])

        ufunc_evalAB = np.frompyfunc(compute_a_b, 4, 1)
        ab_values = np.vstack(ufunc_evalAB(t, h_rad, p1, p2))
        # ab_values = np.array([compute_a_b(h, p1[i], p2[i]) for i, h in enumerate(h_rad)])
        a, b = ab_values[:, 0], ab_values[:, 1]

        # Step 4: Calculate post-adaptation values Ra_, Ga_, Ba_
        Ra_ = (460 * p2 + 451 * a + 288 * b) / 1403.0
        Ga_ = (460 * p2 - 891 * a - 261 * b) / 1403.0
        Ba_ = (460 * p2 - 220 * a - 6300 * b) / 1403.0

        # Step 5: Convert Ra_, Ga_, Ba_ to R_, G_, B_
        def post_adaptation_transform(value_a, FL):
            return np.sign(value_a - 0.1) * (100.0 / FL) * ((27.13 * np.abs(value_a - 0.1)) / (400 - np.abs(value_a - 0.1))) ** (1 / 0.42)

        R_ = post_adaptation_transform(Ra_, params["FL"])
        G_ = post_adaptation_transform(Ga_, params["FL"])
        B_ = post_adaptation_transform(Ba_, params["FL"])

        # Step 6: Calculate Rc, Gc, Bc using inverse Hunt-Pointer-Estevez matrix
        RcGcBc = np.column_stack([R_, G_, B_]).dot(self.inv_Mhpe.T).dot(self.Mcat02.T)

        # Step 7: Adjust Rc, Gc, Bc to R, G, B
        RGB = RcGcBc / np.array([params["Dr"], params["Dg"], params["Db"]])

        # Step 8: Convert R, G, B to XYZ using inverse chromatic adaptation
        XYZ = RGB.dot(self.inv_Mcat02.T)

        return XYZ


    # def rgb_to_jch(self, rgb_image):
    #     """
    #     Converts an RGB image to JCH color space using the CIECAM02 model.

    #     Args:
    #         rgb_image (numpy.ndarray): RGB image array.

    #     Returns:
    #         numpy.ndarray: JCH representation of the image.
    #     """
    #     XYZ_image = rgb2xyz(rgb_image)
    #     JCH_image = self.xyz_to_ciecam02(XYZ_image)
    #     return JCH_image