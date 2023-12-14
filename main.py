import cv2
import numpy as np
from fsm import FSM
import serial
import time
import threading

# Load YOLO model
net = cv2.dnn.readNet("models/yolov3.weights", "models/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

global frame
frame = None

SERIAL_PORT = 'COM5'
BAUD_RATE = 9600
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

drone_fsm = FSM()
drone_fsm.start_video_stream()

# Load class labels
with open("data/coco.names", "r") as f:
    class_labels = f.read().strip().split("\n")

# Thread-safe variables for object detection
detected_object = None
frame_lock = threading.Lock()

def object_detection_thread():
    try:
        global detected_object
        while True:
            if drone_fsm.is_targeting_active:
                with frame_lock:
                    frame_copy = frame.copy() if frame is not None else None

                if frame_copy is not None:
                    frame_copy = resize_frame(frame_copy, scale_percent=75)  # Resize frame
                    frame_copy, _ = detect_and_follow_object(frame_copy, drone_fsm)
                    cv2.imshow("Object Detection", frame_copy)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            time.sleep(0.1)
    except Exception as e:
        print(f"Exception in object detection thread: {e}")

# Resizes the frame based of scale %
def resize_frame(frame, scale_percent=75):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized_frame

# Start the thread right away, but it will only process frames when targeting is active
object_detection_thread = threading.Thread(target=object_detection_thread, daemon=True)
object_detection_thread.start()

# command from the raw data
def extract_command(data):
    try:
        # Decode data to string, ignore errors
        decoded_data = data.decode('utf-8', errors='ignore')
        # Search for the command part
        if 'tl' in decoded_data:
            return 'TL'
        elif '$' in decoded_data:
            return '$'
        elif 'T' in decoded_data in decoded_data:
            return 'T'
        return None
    except UnicodeDecodeError as e:
        print(f"Decode error: {e}")
        return None

# Working First Person Seen Tracking
def detect_and_follow_object(frame, drone_fsm):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    conf_threshold, nms_threshold = 0.5, 0.4
    person_class_id = 0  # Person ID

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id == person_class_id:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = center_x - w // 2, center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    detected_object = None

    for i in indices:
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        # Draw bounding box and calculate center
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center_x, center_y = x + w // 2, y + h // 2

        # Draw label from COCO data
        label = "Person"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Set detected_object for the first detected person
        detected_object = {
            'center_x': center_x,
            'center_y': center_y,
            'width': width,
            'height': height
        }
        break

    follow_object(drone_fsm, detected_object)
    return frame, None

# Working First Object Seen Tracking
# def detect_and_follow_object(frame, drone_fsm):
#     height, width, _ = frame.shape
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)

#     # Initialization
#     class_ids, confidences, boxes = [], [], []
#     conf_threshold, nms_threshold = 0.5, 0.4

#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > conf_threshold:
#                 # Object detected
#                 center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
#                 x, y = center_x - w / 2, center_y - h / 2

#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

#     if len(indices) > 0 and isinstance(indices, tuple):
#         indices = indices[0]  # Access the first element of the tuple

#     for i in indices:
#         print(f"Detected object {i}: Class ID {class_ids[i]}, Confidence {confidences[i]}, Box {boxes[i]}")
#         box = boxes[i]
#         x, y, w, h = box[0], box[1], box[2], box[3]

#         # Convert coordinates to integers
#         x, y, w, h = int(x), int(y), int(w), int(h)

#         # Draw bounding box and calculate center
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         center_x, center_y = x + w // 2, y + h // 2

#         # Get label from class ID
#         label = class_labels[class_ids[i]]
#         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Set detected_object for the first detected object
#         detected_object = {
#         'center_x': x + w // 2,
#         'center_y': y + h // 2,
#         'width': width,
#         'height': height
#     }

#     follow_object(drone_fsm, detected_object)
#     return frame, None

# Forward/Backward, Left/Right, & Up/Down Tracking
# def follow_object(drone_fsm, detected_object):
#     if detected_object is not None:
#         center_x = detected_object['center_x']
#         center_y = detected_object['center_y']
#         frame_width = detected_object['width']
#         frame_height = detected_object['height']

#         # Define ideal bounding box size (width and height) for the desired distance
#         ideal_width = frame_width * 0.5  # Example value, adjust as needed
#         ideal_height = frame_height * 0.5  # Example value, adjust as needed

#         # Threshold for movement
#         size_threshold = 0.3  # Adjust as needed

#         # Check if the person is too close or too far
#         if frame_width > ideal_width * (1 + size_threshold):
#             drone_fsm.tello.move_back(10)  # Move drone backward
#         elif frame_width < ideal_width * (1 - size_threshold):
#             drone_fsm.tello.move_forward(10)  # Move drone forward

#         # Center of the frame
#         frame_center_x, frame_center_y = frame_width / 2, frame_height / 2

#         # Threshold for movement
#         horizontal_threshold = 40  # Adjust as needed for horizontal movement
#         vertical_threshold = 30    # Adjust as needed for vertical movement

#         # Horizontal movement
#         if abs(center_x - frame_center_x) > horizontal_threshold:
#             if center_x < frame_center_x:
#                 drone_fsm.tello.move_left(20)
#             else:
#                 drone_fsm.tello.move_right(20)

#         # Vertical movement
#         if abs(center_y - frame_center_y) > vertical_threshold:
#             if center_y < frame_center_y:
#                 drone_fsm.tello.move_up(20)
#             else:
#                 drone_fsm.tello.move_down(20)
#     else:
#         print("Invalid or incomplete detected_object data")

# Side to Side / Up to Down Tracking only
def follow_object(drone_fsm, detected_object):
    if detected_object is not None:
        center_x = detected_object['center_x']
        center_y = detected_object['center_y']
        frame_width = detected_object['width']
        frame_height = detected_object['height']

        # Center of the frame
        frame_center_x, frame_center_y = frame_width / 2, frame_height / 2

        # Threshold for movement
        horizontal_threshold = 40  # Adjust as needed for horizontal movement
        vertical_threshold = 30    # Adjust as needed for vertical movement

        # Horizontal movement
        if abs(center_x - frame_center_x) > horizontal_threshold:
            if center_x < frame_center_x:
                drone_fsm.tello.move_left(30)
            else:
                drone_fsm.tello.move_right(30)

        # Vertical movement
        if abs(center_y - frame_center_y) > vertical_threshold:
            if center_y < frame_center_y:
                drone_fsm.tello.move_up(30)
            else:
                drone_fsm.tello.move_down(30)
    else:
        print("Invalid or incomplete detected_object data")

try:
    while True:
        with frame_lock:
            frame = drone_fsm.tello.get_frame_read().frame
        if drone_fsm.state == 'Targeting':
            with frame_lock:
                if detected_object is not None:
                    # Calculate frame dimensions
                    frame_height, frame_width = frame.shape[:2]
                    # Logic to follow the detected object
                    follow_object(drone_fsm, detected_object['center_x'], detected_object['center_y'], frame_width, frame_height)

        cv2.imshow("Drone Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if ser.in_waiting:
            data = ser.read(ser.in_waiting)
            command = extract_command(data)
            if command:
                drone_fsm.handle_command(command)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Program terminated by user")

except serial.SerialException:
    print("Serial communication error")

finally:
    cv2.destroyAllWindows()
    drone_fsm.stop_video_stream()
    drone_fsm.cleanup()
    ser.close()