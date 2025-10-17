import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class CameraCalibration():
    """ Class that calibrate camera using chessboard images.

    Attributes:
        mtx (np.array): Camera matrix 
        dist (np.array): Distortion coefficients
    """
    def __init__(self, image_dir, nx, ny, debug=False):
        """ Init CameraCalibration.

        Parameters:
            image_dir (str): path to folder contains chessboard images
            nx (int): width of chessboard (number of squares)
            ny (int): height of chessboard (number of squares)
        """
        fnames = glob.glob("{}/*".format(image_dir))
        
        # --- FIX 1: Ensure calibration images are found ---
        if not fnames:
            raise FileNotFoundError(f"No files found in calibration directory: {image_dir}")

        objpoints = []
        imgpoints = []
        
        # Coordinates of chessboard's corners in 3D
        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        
        # Initialize img to None to handle case where no images are processed
        img = None 
        
        # Go through all chessboard images
        for f in fnames:
            try:
                img = mpimg.imread(f)
            except Exception as e:
                # Handle case where file is found but not readable as an image
                print(f"Warning: Could not read image file {f}: {e}")
                continue

            # Convert to grayscale image
            # NOTE: cv2.cvtColor expects BGR/RGB input. mpimg.imread loads as RGB.
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find chessboard corners
            # NOTE: Usually corners are found on the grayscale image, 'gray'
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            
            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)
        
        # --- FIX 2: Check if any successful corner detection occurred ---
        if not objpoints:
            raise Exception(
                "Unable to calibrate camera: No chessboard corners were successfully found in any image. "
                "Check image quality, chessboard size (nx, ny), and file paths."
            )
            
        # img is guaranteed to hold the last image read due to initialization and loop logic
        shape = (img.shape[1], img.shape[0])
        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

        if not ret:
            raise Exception("Unable to calibrate camera: cv2.calibrateCamera failed.")

    def undistort(self, img):
        """ Return undistort image.

        Parameters:
            img (np.array): Input image

        Returns:
            Image (np.array): Undistorted image
        """
        # Note: No need to convert to grayscale for undistortion
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)