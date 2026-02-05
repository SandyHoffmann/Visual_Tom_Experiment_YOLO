import cv2
import numpy as np

# ! Set the path to your video file and window name
VIDEO_PATH = "VTOM - Vídeos/cenario1.mp4"
WINDOW = "Select Areas (Polygon) - Press ESC to quit"

posicoes_fixas = {}  
current_points = []
typing_name = False
current_name = ""

def draw_polygon(event, x, y, flags, param):
    global current_points, typing_name

    if typing_name:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_points) > 2:
            typing_name = True

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

for _ in range(19):
    ret, _ = cap.read()

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read frame")

frame_copy = frame.copy()
cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, draw_polygon)

print("Instructions:")
print("- Left Click: Add a point")
print("- Right Click: Finish current polygon and type name")
print("- Enter: Confirm name")
print("- ESC: Quit and print results")

while True:
    display_frame = frame_copy.copy()

    for name, pts in posicoes_fixas.items():
        pts_array = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(display_frame, [pts_array], True, (0, 255, 0), 2)
        cv2.putText(display_frame, name, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if len(current_points) > 0:
        pts_array = np.array(current_points, np.int32).reshape((-1, 1, 2))
        is_closed = True if typing_name else False
        color = (0, 255, 255) if typing_name else (0, 0, 255)
        
        cv2.polylines(display_frame, [pts_array], is_closed, color, 2)
        
        for p in current_points:
            cv2.circle(display_frame, p, 3, (0, 0, 255), -1)

    if typing_name:
        cv2.putText(display_frame, f"Name: {current_name}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow(WINDOW, display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27: 
        break
    
    elif typing_name:
        if key == 8: 
            current_name = current_name[:-1]
        elif key == 13:
            if current_name != "" and len(current_points) > 2:
                posicoes_fixas[current_name] = current_points.copy()
                print(f"Added polygon '{current_name}' with {len(current_points)} points.")
            typing_name = False
            current_name = ""
            current_points = []
        elif key != 255:
            current_name += chr(key)

cap.release()
cv2.destroyAllWindows()

print("\nFinal posicoes_fixas dictionary:")
for name, pts in posicoes_fixas.items():
    x_coords = [p[0] for p in pts]
    y_coords = [p[1] for p in pts]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    pts = (min_x, min_y, max_x, max_y)

    print(f"'{name}': {pts},")