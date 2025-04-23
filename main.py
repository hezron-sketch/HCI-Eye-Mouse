import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize camera and face mesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Configuration constants
SCROLL_MARGIN = 0.1  # 10% screen edge margin
LONG_WINK_THRESHOLD = 0.5  # Seconds for right-click
DRAG_HOLD_DURATION = 1.0  # Seconds to initiate drag

# State tracking variables
last_click_time = 0
drag_start_time = 0
dragging = False
wink_start_time = 0
scroll_cooldown = 0

# Instruction text parameters
INSTRUCTION_TEXT = [
    "1. Look at screen position to move cursor",
    "2. Brief left eye close - Left click",
    "3. Hold left eye closed - Right click", 
    "4. Gaze at screen edges - Scroll",
    "5. Maintain closed eye - Drag items"
]
TEXT_COLOR = (0, 255, 0)  # Green
BG_COLOR = (0, 0, 0)        # Black
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
LINE_SPACING = 30

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    
    current_time = time.time()
    
    # Add semi-transparent background for instructions
    y_start = 10
    for line in INSTRUCTION_TEXT:
        text_size = cv2.getTextSize(line, FONT, FONT_SCALE, 1)[0]
        cv2.rectangle(frame, 
                     (5, y_start - 20), 
                     (text_size[0] + 10, y_start + 10), 
                     BG_COLOR, -1)
        y_start += LINE_SPACING

    # Draw instructions
    y_offset = 30
    for line in INSTRUCTION_TEXT:
        cv2.putText(frame, line, (10, y_offset), 
                    FONT, FONT_SCALE, TEXT_COLOR, 1, cv2.LINE_AA)
        y_offset += LINE_SPACING
    
    if landmark_points:
        landmarks = landmark_points[0].landmark
        
        # Gaze tracking (right eye iris)
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                if not dragging:
                    pyautogui.moveTo(screen_x, screen_y, _pause=False)

        # Eye state detection (left eye)
        left_eye = [landmarks[145], landmarks[159]]
        eye_closed = (left_eye[0].y - left_eye[1].y) < 0.004
        
        # Visual feedback for eye state
        eye_status = "CLOSED" if eye_closed else "OPEN"
        cv2.putText(frame, f"Left Eye: {eye_status}", (10, frame_h - 20),
                   FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Wink detection
        if eye_closed:
            if wink_start_time == 0:
                wink_start_time = current_time
        else:
            if wink_start_time > 0:
                wink_duration = current_time - wink_start_time
                if wink_duration < LONG_WINK_THRESHOLD:
                    pyautogui.click()
                else:
                    pyautogui.rightClick()
                wink_start_time = 0

        # Edge-activated scrolling
        if current_time - scroll_cooldown > 0.2:
            if screen_x < SCROLL_MARGIN * screen_w:
                pyautogui.hscroll(-40)
                scroll_cooldown = current_time
            elif screen_x > (1 - SCROLL_MARGIN) * screen_w:
                pyautogui.hscroll(40)
                scroll_cooldown = current_time
            if screen_y < SCROLL_MARGIN * screen_h:
                pyautogui.scroll(40)
                scroll_cooldown = current_time
            elif screen_y > (1 - SCROLL_MARGIN) * screen_h:
                pyautogui.scroll(-40)
                scroll_cooldown = current_time

        # Drag-and-drop functionality
        if eye_closed and not dragging:
            if drag_start_time == 0:
                drag_start_time = current_time
            elif current_time - drag_start_time > DRAG_HOLD_DURATION:
                pyautogui.mouseDown()
                dragging = True
        elif not eye_closed and dragging:
            pyautogui.mouseUp()
            dragging = False
            drag_start_time = 0

    # Display frame
    cv2.imshow('Eye Controlled Mouse', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()