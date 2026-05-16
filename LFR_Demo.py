import cv2
import numpy as np

# Initialize camera (Try 0 if 1 doesn't work)
cap = cv2.VideoCapture(1)
cap.set(3, 160)
cap.set(4, 120)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert BGR to HSV color space for better color isolation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for dark/black colors in HSV
    # Lower bound: Hue=0, Saturation=0, Value=0
    # Upper bound: Hue=180, Saturation=255, Value=50 (Adjust 50 higher if it's too strict)
    low_b = np.array([0, 0, 0], dtype=np.uint8)
    high_b = np.array([180, 255, 50], dtype=np.uint8)
    
    # Create the mask
    mask = cv2.inRange(hsv, low_b, high_b)
    
    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Only draw if at least one contour was found
    if len(contours) > 0:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        
        # Draw the largest contour (Notice [c] is wrapped in a list)
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)
        
    # Display the results
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()