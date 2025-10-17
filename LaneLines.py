import cv2
import numpy as np
import matplotlib.image as mpimg

def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    return np.sum(bottom_half, axis=0)

class LaneLines:
    """ Class containing information about detected lane lines."""

    def __init__(self):
        """Init Lanelines."""
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.clear_visibility = True
        self.dir = []
        
        # Load and normalize direction images
        self.left_curve_img = mpimg.imread('left_turn.png')
        self.right_curve_img = mpimg.imread('right_turn.png')
        self.keep_straight_img = mpimg.imread('straight.png')
        
        self.left_curve_img = cv2.normalize(src=self.left_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.right_curve_img = cv2.normalize(src=self.right_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.keep_straight_img = cv2.normalize(src=self.keep_straight_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # HYPERPARAMETERS
        self.nwindows = 9
        self.margin = 100
        self.minpix = 50

    def forward(self, img):
        """Take a image and detect lane lines."""
        self.extract_features(img)
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height):
        """ Return all pixel that in a specific window."""
        topleft = (center[0]-margin, center[1]-height//2)
        bottomright = (center[0]+margin, center[1]+height//2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx&condy], self.nonzeroy[condx&condy]

    def extract_features(self, img):
        """ Extract features from a binary image."""
        self.img = img
        self.window_height = int(img.shape[0]//self.nwindows)

        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def find_lane_pixels(self, img):
        """Find lane pixels from a binary warped image."""
        assert(len(img.shape) == 2)
        out_img = np.dstack((img, img, img))
        histogram = hist(img)
        midpoint = histogram.shape[0]//2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height//2

        leftx, lefty, rightx, righty = [], [], [], []

        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)

            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)

            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                leftx_current = np.int32(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = np.int32(np.mean(good_right_x))

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img):
        """Find the lane line from an image and draw it."""
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)

        if len(lefty) > 1500:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 1500:
            self.right_fit = np.polyfit(righty, rightx, 2)
            
        if self.left_fit is None or self.right_fit is None:
             print("Warning: Polyfit failed. Not drawing lines on this frame.")
             return np.dstack((img, img, img))

        maxy = img.shape[0] - 1
        miny = img.shape[0] // 3
        if len(lefty):
            maxy = max(maxy, np.max(lefty))
            miny = min(miny, np.min(lefty))
        if len(righty):
            maxy = max(maxy, np.max(righty))
            miny = min(miny, np.min(righty))

        ploty = np.linspace(miny, maxy, img.shape[0])

        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(y)
            cv2.line(out_img, (l, y), (r, y), (0, 255, 0))

        lR, rR, pos = self.measure_curvature()

        return out_img

    def plot(self, out_img):
        """
        Overlays the directional widget, curvature, and position on the image.
        Content is stacked vertically within the widget box.
        """
        np.set_printoptions(precision=6, suppress=True)
        if self.left_fit is None or self.right_fit is None:
            return out_img 

        lR, rR, pos = self.measure_curvature()

        value = self.left_fit[0] if abs(self.left_fit[0]) > abs(self.right_fit[0]) else self.right_fit[0]

        if abs(value) <= 0.00015:
            self.dir.append('F')
        elif value < 0:
            self.dir.append('L')
        else:
            self.dir.append('R')
        
        if len(self.dir) > 10:
            self.dir.pop(0)

        # ------------------------------------------------------------------
        # START: VERTICAL WIDGET CONFIGURATION
        W_IMG, H_IMG = 120, 120    # Direction image size
        TOP_PADDING = 20           # NEW: Padding from the top of the widget to the image
        
        # Calculate box dimensions for proper fit
        W = 280                       # Box width
        H = 300                       # Box height
        
        widget_x, widget_y = 20, 20   # Widget top-left corner
        
        # Image positioning for centering
        img_x_offset = (W - W_IMG) // 2 
        
        # Text positioning setup
        text_x_start = 15             # Text start margin from widget edge
        text_line_height = 30         # Vertical spacing between text lines
        
        # KEY FIX: The vertical start position of the entire content block
        # Image starts at TOP_PADDING (20px)
        # Text starts at TOP_PADDING + H_IMG + 35 (20 + 120 + 35 = 175)
        text_y_start = TOP_PADDING + H_IMG + 35
        
        # Increased Font Sizes for visibility
        FONT_SCALE_MAIN = 0.7
        FONT_SCALE_CURVE = 0.65
        FONT_SCALE_STATUS = 0.65
        # ------------------------------------------------------------------

        # 1. Draw semi-transparent background box
        overlay = out_img.copy()
        cv2.rectangle(overlay, (widget_x, widget_y), (widget_x + W, widget_y + H), (40, 40, 40), -1)
        alpha = 0.6
        out_img[widget_y:widget_y+H, widget_x:widget_x+W] = cv2.addWeighted(
            overlay[widget_y:widget_y+H, widget_x:widget_x+W],
            alpha,
            out_img[widget_y:widget_y+H, widget_x:widget_x+W],
            1 - alpha,
            0
        )
        
        # 2. Prepare Direction Image and Messages
        direction = max(set(self.dir), key = self.dir.count)
        
        if direction == 'L':
            img = cv2.resize(self.left_curve_img, (W_IMG, H_IMG))
            msg = "Left Curve Ahead"
            curvature_msg = "Curvature = {:.0f} m".format(lR)
        elif direction == 'R':
            img = cv2.resize(self.right_curve_img, (W_IMG, H_IMG))
            msg = "Right Curve Ahead"
            curvature_msg = "Curvature = {:.0f} m".format(rR)
        else:
            img = cv2.resize(self.keep_straight_img, (W_IMG, H_IMG))
            msg = "Keep Straight Ahead"
            curvature_msg = ""
            
        # 3. Overlay Direction Image (Centered and padded from top)
        # Image Y-coordinates start at widget_y + TOP_PADDING
        roi_y, roi_h = widget_y + TOP_PADDING, widget_y + TOP_PADDING + H_IMG
        roi_x, roi_w = widget_x + img_x_offset, widget_x + img_x_offset + W_IMG
        
        if img.shape[-1] == 4:
            roi = out_img[roi_y:roi_h, roi_x:roi_w]
            alpha_img = img[..., 3] / 255.0

            for c in range(3):
                roi[..., c] = np.uint8(
                    roi[..., c] * (1 - alpha_img) +  
                    img[..., c] * alpha_img          
                )
            out_img[roi_y:roi_h, roi_x:roi_w] = roi
        else:
            out_img[roi_y:roi_h, roi_x:roi_w] = img 

        # 4. Lane info text (stacked vertically)
        current_text_y = widget_y + text_y_start
        
        # Line 1: Direction Message
        cv2.putText(out_img, msg, (widget_x + text_x_start, current_text_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_MAIN, (255, 255, 255), 2)
        current_text_y += text_line_height
        
        # Line 2: Curvature (only for turns)
        if curvature_msg:
            cv2.putText(out_img, curvature_msg, (widget_x + text_x_start, current_text_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_CURVE, (255, 255, 255), 2)
            current_text_y += text_line_height
        
        # Line 3: Good Lane Keeping status (Fixed green text)
        cv2.putText(
            out_img,
            "Good Lane Keeping",
            org=(widget_x + text_x_start, current_text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=FONT_SCALE_STATUS,
            color=(0, 255, 0),
            thickness=2)
        current_text_y += text_line_height

        # Line 4: Vehicle Position (Part 1)
        cv2.putText(
            out_img,
            "Vehicle is {:.2f} m".format(pos),
            org=(widget_x + text_x_start, current_text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=FONT_SCALE_STATUS,
            color=(255, 255, 255),
            thickness=2)
        current_text_y += (text_line_height // 2)

        # Line 5: Vehicle Position (Part 2: "away from center")
        cv2.putText(
            out_img,
            "away from center",
            org=(widget_x + text_x_start, current_text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=FONT_SCALE_STATUS,
            color=(255, 255, 255),
            thickness=2)


        return out_img

    def measure_curvature(self):
        ym = 30/720
        xm = 3.7/700

        left_fit = self.left_fit.copy()
        right_fit = self.right_fit.copy()
        y_eval = 700 * ym

        # Compute R_curve (radius of curvature)
        left_curveR =  ((1 + (2*left_fit[0] *y_eval + left_fit[1])**2)**1.5)  / np.absolute(2*left_fit[0])
        right_curveR = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0]) if np.absolute(2*right_fit[0]) > 0 else 99999.0

        xl = np.dot(self.left_fit, [700**2, 700, 1])
        xr = np.dot(self.right_fit, [700**2, 700, 1])
        pos = (1280//2 - (xl+xr)//2)*xm
        return left_curveR, right_curveR, pos