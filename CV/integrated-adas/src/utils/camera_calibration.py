class CameraCalibration:
    """ Class that calibrates the camera using chessboard images.

    Attributes:
        mtx (np.array): Camera matrix 
        dist (np.array): Distortion coefficients
    """
    
    def __init__(self, image_dir, nx, ny):
        """ Initializes the CameraCalibration class.

        Parameters:
            image_dir (str): Path to folder containing chessboard images
            nx (int): Width of chessboard (number of squares)
            ny (int): Height of chessboard (number of squares)
        """
        import numpy as np
        import cv2
        import glob
        import matplotlib.image as mpimg

        fnames = glob.glob(f"{image_dir}/*")
        
        if not fnames:
            raise FileNotFoundError(f"No files found in calibration directory: {image_dir}")

        objpoints = []
        imgpoints = []
        
        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        
        for f in fnames:
            try:
                img = mpimg.imread(f)
            except Exception as e:
                print(f"Warning: Could not read image file {f}: {e}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            
            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)
        
        if not objpoints:
            raise Exception("Unable to calibrate camera: No chessboard corners were successfully found in any image.")

        shape = (img.shape[1], img.shape[0])
        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

        if not ret:
            raise Exception("Unable to calibrate camera: cv2.calibrateCamera failed.")

    def undistort(self, img):
        """ Returns the undistorted image.

        Parameters:
            img (np.array): Input image

        Returns:
            Image (np.array): Undistorted image
        """
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)