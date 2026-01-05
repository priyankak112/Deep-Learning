 🚁 Drone-Based Disaster Rescue System using YOLOv8 

 <img width="1920" height="882" alt="output_image" src="https://github.com/user-attachments/assets/bbd9ec66-bd12-4d55-b7f3-1b2b53831719" /> 
 
 📌 Overview
This project is an AI-powered disaster rescue system that detects people in drone-captured images and videos using YOLOv8. It helps rescue teams quickly identify survivors in disaster-affected areas.

🎯 Objectives
- Detect people in aerial drone footage
- Count number of detected persons
- Trigger rescue alerts for continuous detection
- Provide a real-time web-based interface

 🛠️ Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Supervision
- Streamlit

 📂 Features
- Image & video input support
- Real-time person detection
- Person counting
- Alert system
- Adjustable detection resolution

 🔁 Workflow
1. Upload drone image or video
2. YOLOv8 performs person detection
3. Bounding boxes and confidence displayed
4. Person count calculated
5. Rescue alert triggered if condition met

 ▶️ How to Run
```bash
pip install ultralytics streamlit opencv-python supervision
streamlit run app.py


















