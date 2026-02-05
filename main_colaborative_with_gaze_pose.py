import cv2
import numpy as np
import os
import json
from collections import defaultdict
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import google.generativeai as genai

# ! Configure your Gemini API key in the environment variable "GEMINI_API_KEY" before running this code.
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini = genai.GenerativeModel("gemini-2.5-flash") 

combinations = {
    "cenario1": {
        "object_positions": {
            "fridge": (330, 91, 464, 330),
            "cabinet1": (452, 55, 543, 145),
            "cabinet2": (541, 54, 630, 148),
            "cabinet3": (629, 52, 715, 145),
            "cabinet4": (714, 50, 806, 148),
            "stove": (798, 194, 926, 317),
            "microwave": (450, 169, 530, 215),
        },
        "object_items": {
            "fridge": ["apple", "lemon cake", "plate", "dish bowl", "cheese", "lettuce", "milk"],
            "cabinet3": ["water glass", "dish bowl", "wine glass"],
            "cabinet1": ["wine","bottle"],
            "cabinet2": ["mayonnaise"],
            "microwave": ["chocolate cake"],
            "stove": ["cupcake", "bread"],
        }
    },
    "cenario2": {
        "object_positions": {
            "cabinet1": (238, 49, 321, 133),
            "cabinet2": (306, 49, 396, 135),
            "cabinet3": (387, 55, 470, 131),
            "cabinet4": (468, 53, 550, 132),
            "stove": (548, 180, 653, 315),
            "cabinet5": (652, 54, 736, 131),
            "cabinet6": (736, 54, 813, 130),
            "cabinet7": (812, 57, 891, 131),
            "cabinet8": (883, 56, 972, 134),
            "fridge": (959, 85, 1089, 293),
            "microwave": (235, 161, 318, 202),
            "dishwasher": (752, 200, 859, 306)
        },
        "object_items": {
            "fridge": ["eggs", "bacon", "ice", "butter"],
            "cabinet3": ["flour", "sugar", "salt"],
            "cabinet1": ["milk"],
            "cabinet5": ["bread"],
            "cabinet8": ["mayonnaise", "cheese", "lettuce"],
            "microwave": ["butter"],
            "stove": ["cupcake", "hot dog"],
        }
    }    
}

# ! Change the scenario here:

atual_scenario = "cenario2"

object_positions = combinations[atual_scenario]["object_positions"]
object_items = combinations[atual_scenario]["object_items"]

# ! You can modify the user beliefs here, for whatever you want to test.

'''beliefs = [
    "believes(user, at(cabinet1, \"wine\"))",
]'''

beliefs = [
    "believes(user, at(fridge, \"eggs\"))",
    "believes(user, at(microwave, \"butter\"))",
]

# ! These requisited items are for debugging and showing where they are located.

# requisited_items = ["wine", "wine glass"] 

# requisited_items = ["bread", "mayonnaise", "cheese", "lettuce"]

requisited_items = ["butter", "eggs", "flour", "sugar", "salt", "milk"] 

# ! Modify questions at the end of the code.

VIDEO_PATH = f'VTOM - Vídeos/cenario{atual_scenario[-1]}.mp4'
WINDOW = "Kitchen Assistant"
TOUCH_FRAMES_REQUIRED = 3
MIN_HAND_SPEED = 2.0
INTERACTION_DEPTH = 0.45
MAX_MISSING = 30

# Initialize Models
yolo_model = YOLO("yolov8n.pt")

# ? I combine pose and face models to improve robustness in case of occlusions

pose_landmarker = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
)

face_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
)

belief_pos = None
belief_gaze = None
missing_frames = 0
touch_counter = defaultdict(int)
prev_wrist_pos = {"L": None, "R": None}

def mp_image(frame):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def to_pixel(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def interaction_zone(bbox):
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    return (x1, y2 - int(h * INTERACTION_DEPTH), x2, y2)

def wrist_speed(curr, prev):
    if curr is None or prev is None: return 0.0
    return np.linalg.norm(np.array(curr) - np.array(prev))

def relative_position(bbox, human_pos):
    ox1, oy1, ox2, oy2 = bbox
    cx, cy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
    dx, dy = cx - human_pos[0], cy - human_pos[1]
    horiz = "left" if dx < -50 else "right" if dx > 50 else "center"
    #depth = "in front" if dy < -50 else "behind" if dy > 50 else "same depth"
    # TODO keeped in front as default for now, future work will improve depth estimation
    depth = "in front"

    return f"{depth} and to the {horiz}"

def build_world_state(looked_object, touched_object):
    # TODO future work will include looked_object and touched_object in the world state
    world = {
        "human": {
            "position_px": belief_pos,
            #"looking_at": looked_object,
            #"touching": touched_object,
        },
        "objects": {},
    }
    if belief_pos:
        for obj, bbox in object_positions.items():
            world["objects"][obj] = {
                "contains": object_items.get(obj, []),
                "relative_to_human": relative_position(bbox, belief_pos),
            }
    return world

def ask_gemini(world, request, beliefs):
    prompt = f"""
    You are an assistive AI in a kitchen.
    World state: {json.dumps(world, indent=2)}
    User request: "{request}"
    User beliefs: {', '.join(beliefs)}
    Explain clearly where the requested items are based on the human's current position (refer to the container as labeled in the world state), but do not mention items that the user already believes to be at a certain location, as long as the location is correct.
    """
    return gemini.generate_content(prompt).text.strip()

cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    looked_object = None
    touched_object = None

    results = yolo_model(frame, classes=[0], verbose=False)
    boxes = [tuple(map(int, b.xyxy[0])) for r in results for b in r.boxes]

    if boxes:
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        px1, py1, px2, py2 = boxes[0]
        crop = frame[py1:py2, px1:px2]
        m_img = mp_image(crop)

        pose = pose_landmarker.detect(m_img)
        if pose.pose_landmarks:
            lm = pose.pose_landmarks[0]
            l_sh = to_pixel(lm[11], px2 - px1, py2 - py1)
            r_sh = to_pixel(lm[12], px2 - px1, py2 - py1)
            nose = to_pixel(lm[0], px2 - px1, py2 - py1)
            
            l_sh = (l_sh[0] + px1, l_sh[1] + py1)
            r_sh = (r_sh[0] + px1, r_sh[1] + py1)
            nose = (nose[0] + px1, nose[1] + py1)

            belief_pos = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2)
            belief_gaze = np.array([nose[0] - belief_pos[0], nose[1] - belief_pos[1]])
            missing_frames = 0

            # TODO touched logic still in progress, is not considered in the final response yet
            for side, (w_idx, e_idx) in {"L": (15, 13), "R": (16, 14)}.items():
                wrist = to_pixel(lm[w_idx], px2 - px1, py2 - py1)
                wrist = (wrist[0] + px1, wrist[1] + py1)
                speed = wrist_speed(wrist, prev_wrist_pos[side])
                prev_wrist_pos[side] = wrist

                for obj, bbox in object_positions.items():
                    ix1, iy1, ix2, iy2 = interaction_zone(bbox)
                    if (ix1 <= wrist[0] <= ix2 and iy1 <= wrist[1] <= iy2) and speed > MIN_HAND_SPEED:
                        touch_counter[obj] += 1
                    else:
                        touch_counter[obj] = max(0, touch_counter[obj] - 1)
                    
                    if touch_counter[obj] >= TOUCH_FRAMES_REQUIRED:
                        touched_object = obj
        else:
            face = face_landmarker.detect(m_img)
            if face.face_landmarks:
                f_lm = face.face_landmarks[0]
                l_eye = to_pixel(f_lm[33], px2 - px1, py2 - py1)
                r_eye = to_pixel(f_lm[263], px2 - px1, py2 - py1)
                nose_tip = to_pixel(f_lm[1], px2 - px1, py2 - py1)
                
                eye_center = ((l_eye[0] + r_eye[0]) // 2 + px1, (l_eye[1] + r_eye[1]) // 2 + py1)
                belief_pos = belief_pos if belief_pos else eye_center
                belief_gaze = np.array([(nose_tip[0] + px1) - eye_center[0], (nose_tip[1] + py1) - eye_center[1]])
                missing_frames = 0
    else:
        missing_frames += 1
        if missing_frames > MAX_MISSING:
            belief_pos = belief_gaze = None

    # TODO Gaze Logic (still in process)
    if belief_pos and belief_gaze is not None:
        gaze_tip = (int(belief_pos[0] + belief_gaze[0] * 2), int(belief_pos[1] + belief_gaze[1] * 2))
        cv2.arrowedLine(frame, belief_pos, gaze_tip, (255, 255, 0), 2)
        for obj, (ox1, oy1, ox2, oy2) in object_positions.items():
            if ox1 <= gaze_tip[0] <= ox2 and oy1 <= gaze_tip[1] <= oy2:
                looked_object = obj

    # Drawing Logic
    for obj, bbox in object_positions.items():
        color = (0, 255, 0)
        if obj == looked_object: color = (0, 0, 255)
        if obj == touched_object: color = (0, 255, 255)
        cv2.rectangle(frame, bbox[:2], bbox[2:], color, 2)
        cv2.putText(frame, obj, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        objetos_lista = object_items.get(obj, [])
        if (objetos_lista):
            objetos_filtrados = [item for item in objetos_lista if item in requisited_items]
            if objetos_filtrados:
                cv2.putText(frame, ", ".join(objetos_filtrados), (bbox[0], bbox[3]- 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") and belief_pos:
        # ! You can modify the request here, to test different scenarios. Just make sure it matches the requisited_items list in terms of what is being asked for.
        # request = "The user intends to drink wine. Indicate where the wine and necessary utensils are located relative to your position"
        # request = "The user intends to eat a sandwich. Indicate where all possible ingredients are located relative to your position"
        request = "The user intends to make pancakes. Indicate where all possible ingredients are located relative to your position"
        world = build_world_state(looked_object, touched_object)
        print("\n[AI Response]:", ask_gemini(world, request, beliefs))
        # wait for user to click w to continue
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == ord("w"): break

    cv2.imshow(WINDOW, frame)
    if key == 27: break

cap.release()
cv2.destroyAllWindows()