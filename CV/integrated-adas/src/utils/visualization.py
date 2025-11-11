def overlay_results(image, lane_lines=None, signs=None, drowsiness_status=None, emotion=None):
    if lane_lines is not None:
        for line in lane_lines:
            # Assuming line is a tuple of (x1, y1, x2, y2)
            cv2.line(image, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)

    if signs is not None:
        for sign in signs:
            # Assuming sign is a dictionary with 'position' and 'label'
            cv2.putText(image, sign['label'], sign['position'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if drowsiness_status is not None:
        status_text = "Drowsiness Detected" if drowsiness_status else "Alert"
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if emotion is not None:
        cv2.putText(image, f"Emotion: {emotion}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return image