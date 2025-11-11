def preprocess_image(image):
    # Resize the image to a standard size
    resized_image = cv2.resize(image, (224, 224))
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Normalize the image
    normalized_image = gray_image / 255.0
    # Expand dimensions to match model input
    processed_image = np.expand_dims(normalized_image, axis=0)
    return processed_image

def overlay_results(image, results):
    for result in results:
        # Draw bounding boxes or labels on the image
        cv2.rectangle(image, (result['x'], result['y']), 
                      (result['x'] + result['width'], result['y'] + result['height']), 
                      (0, 255, 0), 2)
        cv2.putText(image, result['label'], (result['x'], result['y'] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image