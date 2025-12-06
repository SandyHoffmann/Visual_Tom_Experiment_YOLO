import os
import cv2
import json
import math
import numpy as np
import supervision as sv
import ollama
from tqdm import tqdm
import sys


CUSTOM_CLIP_DIR = "E:/Python/" 
sys.path.insert(0, CUSTOM_CLIP_DIR)

from ultralytics import YOLOWorld

VIDEOS_FOLDER = "VTOM - Vídeos"
QUESTIONS_FILE = "VTOM - Vídeos/questions.json"
OUTPUT_FILE = "vtom_results_hybrid.json"
MODEL_ID = "yolov8x-worldv2.pt"
CONFIDENCE_THRESHOLD = 0.2 
IOU_THRESHOLD = 0.4          
MOVEMENT_THRESHOLD = 20     

CLASSES = [
    "Person",
    "apple",
    "bag of chips",
    "book",
    "bottle of wine",
    "condiment bottle",
    "cupcake",
    "dish bowl",
    "plate",
    "remote control",
    "salmon",
    "water glass",
    "wine glass",
    "cabinet",
    "coffee table",
    "desk",
    "sofa",
    "fridge",
    "refrigerator",
    "microwave",
    "stove",
    "oven",
    "dishwasher",
    "kitchen table",
    "bed",
    # "shelf",
    "bag of chips"

]

tracked_objects_history = {} 
next_persistent_id = 0


def reset_tracker_state():
    global tracked_objects_history, next_persistent_id
    tracked_objects_history = {}
    next_persistent_id = 0

def map_detections_to_history(detections):
    """
    Maps current detections to historical tracked objects using IoU for persistent IDs.
    Args:
        detections (sv.Detections): Current frame detections.

    Returns:
        sv.Detections: Detections with assigned persistent tracker IDs.
    """
    global tracked_objects_history, next_persistent_id
    
    current_bboxes = detections.xyxy
    assigned_ids = []
    
    # * Get history IDs
    history_ids = list(tracked_objects_history.keys())
    
    if not history_ids:
        for _ in range(len(current_bboxes)):
            assigned_ids.append(next_persistent_id)
            next_persistent_id += 1
    else:
        history_bboxes = np.array([tracked_objects_history[id]['bbox'] for id in history_ids])
        
        # * Check IoU between current and history to assign IDs.
        iou_matrix = sv.box_iou_batch(current_bboxes, history_bboxes)
        
        for i in range(len(current_bboxes)):
            best_match_index = iou_matrix[i].argmax()
            best_iou = iou_matrix[i][best_match_index]
            
            # ? If IoU is above threshold, assign historical ID to current detection, with suppression to avoid multiple assignments.
            if best_iou >= IOU_THRESHOLD:
                matched_id = history_ids[best_match_index]
                assigned_ids.append(matched_id)
                iou_matrix[:, best_match_index] = -1.0 # ? Suppress this history ID to prevent multiple assignments to the same object
            else:
                assigned_ids.append(next_persistent_id)
                next_persistent_id += 1

    # * Update tracked_objects_history with current detections
    for i, persistent_id in enumerate(assigned_ids):
        x1, y1, x2, y2 = current_bboxes[i]
        # ? Calculate center point of the bounding box
        center = {'x': (x1 + x2) / 2, 'y': (y1 + y2) / 2}
        tracked_objects_history[persistent_id] = {
            'bbox': current_bboxes[i],
            'center': center,
            'last_seen_frame': -1 
        }
        
    detections.tracker_id = np.array(assigned_ids)
    return detections

def is_agent_present(detections, classes):
    """
    Checks if the agent (Person) is present in the current detections (a simple approach).
    
    Args:
        detections (sv.Detections): Current frame detections.
        classes (list): List of class names corresponding to model outputs.
    Returns:
        bool: True if agent is present, False otherwise.
        int or None: Tracker ID of the agent if present, None otherwise.
    """

    agent_keywords = ["Person"]
    for i, class_id in enumerate(detections.class_id):
        class_name = classes[class_id]
        if class_name in agent_keywords:
            return True, detections.tracker_id[i]
    return False, None


def analyze_video_events(video_path, model, classes):
    """
    Analyzing video to log events based on object detections and tracking.
    Args:
        video_path (str): Path to the video file.
        model (YOLOWorld): Pre-loaded YOLOWorld model for inference.
        classes (list): List of class names corresponding to model outputs.

    Returns:
        list: Event log containing frame-wise object presence and agent status.

    """
    global next_persistent_id
    reset_tracker_state()
    
    video_info = sv.VideoInfo.from_video_path(video_path)
    frame_generator = sv.get_video_frames_generator(video_path)
    
    event_log = [] 
    frame_interval = 5 
    
    print(f"Processing {os.path.basename(video_path)}...")
    
    for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):

        if frame_idx % frame_interval != 0:
            continue
            
        # * Infering detections by passing the frame through the YOLOWorld model
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD)

        # * This part converts the inference results into an object containing bounding boxes, class IDs, and confidence scores.
        # ? With NMS (Non-Maximum Suppression) to filter overlapping boxes
        detections = sv.Detections.from_ultralytics(results[0]).with_nms(threshold=0.1)
        
        # * mapping current detections to historical tracked objects for persistent IDs

        detections = map_detections_to_history(detections)

        # * Check if the agent is present in the current frame
        agent_is_in_scene, agent_id = is_agent_present(detections, classes)
        
        # * Get agent center if present for distance calculations
        agent_center = None
        if agent_is_in_scene and agent_id in tracked_objects_history:
            agent_center = tracked_objects_history[agent_id]['center']

        current_objects_in_frame = []
        

        for i, tracker_id in enumerate(detections.tracker_id):
            raw_class_name = classes[detections.class_id[i]]
            
            final_name = raw_class_name
            
            if final_name in ["Person"]:
                continue
            
            distance_str = ""
            if agent_center is not None:
                # ? Gets center of the object and calculates distance to agent using Euclidean distance. 
                # ? It doesn't account for depth, just 2D pixel distance (x, y).
                obj_center = tracked_objects_history[tracker_id]['center']
                dx = obj_center['x'] - agent_center['x']
                dy = obj_center['y'] - agent_center['y']
                distance = math.sqrt(dx**2 + dy**2)
                distance_str = f" | DistToAgent: {int(distance)}px"
                tracked_objects_history[tracker_id]['distance'] = distance
            else:
                distance_str = " | Agent ABSENT"
            


            current_objects_in_frame.append(f"{final_name} (ID:{tracker_id}){distance_str}")

                
        show_frame = visualize_detections(frame, detections, classes)
        cv2.imshow("Detections", show_frame)
        cv2.waitKey(1)


        state_entry = {
            "frame": frame_idx,
            "timestamp": f"{frame_idx / video_info.fps:.2f}s",
            "agent_present": agent_is_in_scene,
            "visible_objects": current_objects_in_frame
        }
        event_log.append(state_entry)        
    return event_log


def visualize_detections(frame, detections, classes):

    """
    Visualizes detections on the frame with bounding boxes and labels.
    Args:
        frame (np.ndarray): The original video frame.
        detections (sv.Detections): Detections to visualize.
        classes (list): List of class names corresponding to model outputs.
    Returns:
        np.ndarray: Annotated frame with visualized detections.
    """

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    labels = []
    for i in range(len(detections)):
        class_id = detections.class_id[i]
        tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else "N/A"
        confidence = detections.confidence[i] if detections.confidence is not None else 0.0
        distance = tracked_objects_history[tracker_id]['distance'] if tracker_id in tracked_objects_history and 'distance' in tracked_objects_history[tracker_id] else None
        class_name = classes[class_id]
        
        labels.append(f"{class_name} - {f'{int(distance)}px' if distance is not None else 'N/A'}")
        # labels.append(f"{class_name} #{tracker_id} ({confidence:.2f}) - {f'{int(distance)}px' if distance is not None else 'N/A'}")

    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, 
        detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, 
        detections=detections, 
        labels=labels
    )

    return annotated_frame


def main():

    print("Loading YOLOWorld...")
    model = YOLOWorld(MODEL_ID)
    model.set_classes(CLASSES)
    
    dict_video_questions = []

    debug = True

    if (not debug):
    
        if os.path.exists(QUESTIONS_FILE):
            with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                dict_video_questions = raw_data 
        else:
            print("Questions file not found. Please check path.")
            return

        for item in dict_video_questions:
            episode = item.get('episode', 'unknown')
            video_path = os.path.join(VIDEOS_FOLDER, f"task_{episode}", "Action_normal.mp4")
            
            if not os.path.exists(video_path):
                print(f"Skipping {video_path}")
                continue
                
            print(f"--- Analyzing Episode {episode} ---")
            event_log = analyze_video_events(video_path, model, CLASSES)

            with open(f"event_log_episode_{episode}.json", 'w', encoding='utf-8') as f:
                json.dump({
                    "episode": episode,
                    "event_log": event_log
                }, f, indent=4)
    else:
        video_path = "VTOM - Vídeos/task_1/Action_normal.mp4"
        event_log = analyze_video_events(video_path, model, CLASSES)

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                "video": os.path.basename(video_path),
                "event_log": event_log
            }, f, indent=4)

        

if __name__ == "__main__":
    main()