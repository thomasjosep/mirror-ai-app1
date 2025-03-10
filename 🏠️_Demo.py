import os
import streamlit as st
from process_frame import ProcessFrame

st.title('AI Fitness Trainer: Squats Analysis')

# Initialize ProcessFrame with thresholds and flip_frame
thresholds = {
    'OFFSET_THRESH': 10,
    'INACTIVE_THRESH': 5,
    'HIP_KNEE_VERT': {
        'NORMAL': [0, 30],
        'TRANS': [30, 60],
        'PASS': [60, 90]
    },
    'HIP_THRESH': [10, 20],
    'KNEE_THRESH': [10, 20, 30],
    'ANKLE_THRESH': 10,
    'CNT_FRAME_THRESH': 5
}
process_frame = ProcessFrame(thresholds, flip_frame=False)

# Check Firestore connection
if process_frame.check_db_connection():
    st.success("Database connection is successful.")
else:
    st.error("Database connection failed.")

recorded_file = 'output_sample.mp4'
sample_vid = st.empty()
sample_vid.video(recorded_file)