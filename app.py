import streamlit as st
import cv2
import time
import tempfile
import numpy as np
from ultralytics import YOLO
import supervision as sv
from PIL import Image

# ================= STREAMLIT UI =================
st.set_page_config(
    page_title="Drone-Based Disaster Rescue System",
    layout="wide"
)

st.title("Drone-Based Disaster Rescue System using YOLOv8")
st.markdown(
    """
    Detect survivors in **drone footage** (video or image) using YOLOv8.
    """
)

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Rescue Alert Settings")

MAX_ALLOWED_PERSONS = st.sidebar.slider(
    "Minimum persons to trigger rescue",
    min_value=1,
    max_value=10,
    value=1
)

ALERT_DURATION = st.sidebar.slider(
    "Detection duration (seconds)",
    min_value=1,
    max_value=10,
    value=3
)

uploaded_file = st.file_uploader(
    " Upload Drone Video or Image",
    type=["mp4", "avi", "mov", "jpg", "png"]
)

# ================= MAIN LOGIC =================
if uploaded_file is not None:

    file_ext = uploaded_file.name.split('.')[-1].lower()
    frame_placeholder = st.empty()
    prev_time = time.time()
    crowd_start_time = None
    rescue_alert = False
    PERSON_CLASS_ID = 0

    model = YOLO("yolov8m.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color=sv.Color(0, 255, 0)
    )

    label_annotator = sv.LabelAnnotator(
        text_scale=0.6,
        text_thickness=1,
        text_color=sv.Color.WHITE,
        text_padding=1
    )

    if file_ext in ["mp4", "avi", "mov"]:
        # ================= VIDEO =================
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for faster processing
            #frame = cv2.resize(frame, (640, 360))

            # YOLO Inference
            result = model(frame, agnostic_nms=True)[0]
            detections = sv.Detections.from_ultralytics(result)

            person_mask = detections.class_id == PERSON_CLASS_ID
            detections = detections[person_mask]
            person_count = len(detections)

            labels = [
                f"Survivor {detections.confidence[i]:.2f}"
                for i in range(person_count)
            ]

            frame = box_annotator.annotate(frame, detections)
            frame = label_annotator.annotate(frame, detections, labels=labels)

            # ================= RESCUE ALERT LOGIC =================
            if person_count >= MAX_ALLOWED_PERSONS:
                if crowd_start_time is None:
                    crowd_start_time = time.time()
                elif time.time() - crowd_start_time >= ALERT_DURATION:
                    rescue_alert = True
            else:
                crowd_start_time = None
                rescue_alert = False

            # ================= ALERT DISPLAY =================
            if rescue_alert:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    " RESCUE ALERT: SURVIVORS DETECTED!",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            # ================= INFO TEXT =================
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            cv2.putText(frame, f"PEOPLE COUNT: {person_count}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb)

        cap.release()
        st.success("✅ Video processing completed")

    elif file_ext in ["jpg", "png"]:
        # ================= IMAGE =================
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)
        #frame = cv2.resize(frame, (640, 360))

        # YOLO Inference
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)

        person_mask = detections.class_id == PERSON_CLASS_ID
        detections = detections[person_mask]
        person_count = len(detections)

        labels = [
            f"Survivor {detections.confidence[i]:.2f}"
            for i in range(person_count)
        ]

        frame = box_annotator.annotate(frame, detections)
        frame = label_annotator.annotate(frame, detections, labels=labels)

        # ALERT if person_count >= MAX_ALLOWED_PERSONS
        if person_count >= MAX_ALLOWED_PERSONS:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 255), -1)
            cv2.putText(
                frame,
                "RESCUE ALERT: SURVIVORS DETECTED!",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            # ================= PERSON COUNT DISPLAY =================
            cv2.putText(
                frame,
                f"PEOPLE COUNT: {person_count}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb)
        st.success("✅ Image processing completed")
