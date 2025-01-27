import cv2
import time
import os

def analyze_video_file(video_path, roi, threshold, work_criteria):
    """
    Analyzes employee activity from a video file.

    Args:
        video_path: Path to the video file.
        roi: Region of interest (tuple: (top, bottom, left, right)).
        threshold: Motion detection threshold.
        work_criteria: Criteria for defining "work" (e.g., duration, intensity).

    Returns:
        Analysis results (e.g., work duration, performance category).
    """

    # 1. Input Selection and Capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None

    # 2. Motion Detection and Analysis
    fgbg = cv2.createBackgroundSubtractorMOG2()
    frame_count = 0
    work_frames = 0 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract ROI
        roi_frame = frame[roi[0]:roi[1], roi[2]:roi[3]]

        # Apply background subtraction
        fgmask = fgbg.apply(roi_frame)

        # Count non-zero pixels in the mask
        num_foreground_pixels = cv2.countNonZero(fgmask)

        # Determine if work is being performed (adjust criteria as needed)
        if num_foreground_pixels > threshold:
            work_frames += 1

        frame_count += 1

    # Calculate work duration (assuming 30 FPS)
    if frame_count > 0:
        work_duration = (work_frames / frame_count) * 30  # seconds

    # 3. Performance Classification
    performance_category = "Poor"
    if work_duration > work_criteria["min_work_duration"]:
        performance_category = "Good"
    if work_duration > work_criteria["min_work_duration"] * 1.5:
        performance_category = "Better"
    if work_duration > work_criteria["min_work_duration"] * 2:
        performance_category = "Best"

    # 4. Visualization and Reporting
    print(f"Work Duration: {work_duration:.2f} seconds")
    print(f"Performance Category: {performance_category}")

    cap.release()

    return {"work_duration": work_duration, "performance": performance_category}

if __name__ == "__main__":
    # Example Usage
    video_path = "C:/Users/Dell/Downloads/doi.mp4" # Replace with the actual video path
    roi = (200, 500, 100, 300) 
    threshold = 50
    work_criteria = {
        "min_work_duration": 60,  # Minimum work duration in seconds
    }

    # Analyze employee activity from the video file
    analysis_results = analyze_video_file(video_path, roi, threshold, work_criteria)