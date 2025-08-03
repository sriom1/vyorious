import streamlit as st
from detections import ObjectDetector
import cv2
from PIL import Image
import numpy as np
import time

def main():
    st.title("Zero-Shot Object Detection")
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = ObjectDetector(enable_logging=True, log_format="csv")
    
    # Sidebar for prompts
    st.sidebar.header("Detection Prompts")
    current_prompts = ", ".join(st.session_state.detector.prompts)
    new_prompts = st.sidebar.text_area("Edit prompts (comma-separated)", value=current_prompts)
    
    if st.sidebar.button("Update Prompts"):
        prompts = [p.strip() for p in new_prompts.split(",") if p.strip()]
        st.session_state.detector.prompts = prompts
        st.success("Prompts updated successfully!")

    # File uploader for prompts
    uploaded_file = st.sidebar.file_uploader("Load prompts from file", type="txt")
    if uploaded_file is not None:
        content = uploaded_file.read().decode()
        prompts = [p.strip() for p in content.split(",") if p.strip()]
        st.session_state.detector.prompts = prompts
        st.sidebar.success("Prompts loaded from file!")

    # Main content - Video capture
    st.header("Live Detection")
    frame_placeholder = st.empty()
    fps_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video frame")
                break

            # Efficient preprocessing: convert only once
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # If your detector expects RGB numpy array, pass frame_rgb directly
            # If it expects a tensor, convert here:
            # frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0

            # Run detection (ensure detect() is optimized for numpy input)
            detections = st.session_state.detector.detect(frame_rgb)

            # Draw detections (on frame_rgb for display)
            for box, score, label in zip(detections["boxes"], detections["scores"], detections["labels"]):
                if score > 0.2:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    label_text = st.session_state.detector.prompts[label]
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_rgb, f"{label_text} {score:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame (avoid PIL conversion, use numpy directly)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            fps_placeholder.text(f"FPS: {fps:.1f}")

            # Optional: break after some time or add a stop button in UI

    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        cap.release()

if __name__ == "__main__":
    main()