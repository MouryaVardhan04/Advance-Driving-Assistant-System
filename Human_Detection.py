import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

class HumanDetector:
    def __init__(self):
        # Initialize HOG descriptor with default people detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Detection parameters (optimized for speed)
        self.winStride = (8, 8)    # Increases step size of sliding window
        self.padding = (4, 4)      # Reduces padding
        self.scale = 1.05          # Increases scaling factor
        self.overlapThresh = 0.65  # NMS threshold
        self.color = (139, 34, 104)  # Purple color for boxes
        
    def detect(self, frame):
        """
        Detect humans in the frame.
        Returns: processed boxes after NMS, original confidences
        """
        # Detect humans using HOG
        rects, weights = self.hog.detectMultiScale(
            frame,
            winStride=self.winStride,
            padding=self.padding,
            scale=self.scale
        )
        
        if len(rects) == 0:
            return [], []
            
        # Convert to x1,y1,x2,y2 format for NMS
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        
        # Apply Non-Max Suppression
        pick = non_max_suppression(boxes, probs=None, overlapThresh=self.overlapThresh)
        
        # Match the picked boxes with their original confidences
        filtered_weights = []
        if len(pick) > 0:
            for picked_box in pick:
                # Find matching original box
                for orig_box, weight in zip(boxes, weights):
                    if np.array_equal(picked_box, orig_box):
                        filtered_weights.append(float(weight))
                        break
        
        return pick, filtered_weights
    
    def draw_detections(self, frame, boxes, confidences):
        """Draw bounding boxes and labels for detected humans"""
        annotated_frame = frame.copy()
        person_count = 0
        
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            person_count += 1
            conf = confidences[i] if i < len(confidences) else 0.0
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.color, 2)
            
            # Draw label background
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x2, y1), self.color, -1)
            
            # Draw person label and confidence
            label = f'P{person_count}: {conf:.2f}'
            cv2.putText(annotated_frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add total count
        if person_count > 0:
            cv2.putText(annotated_frame, f'Total Persons: {person_count}',
                       (20, 450), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated_frame

# Maintain backward compatibility with old code
def Detector(frame):
    """Legacy function for backward compatibility"""
    detector = HumanDetector()
    boxes, weights = detector.detect(frame)
    return detector.draw_detections(frame, boxes, weights)