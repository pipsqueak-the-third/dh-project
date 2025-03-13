import marimo

__generated_with = "0.11.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    from ultralytics import YOLO
    return YOLO, cv2, mdates, mo, np, plt


@app.cell
def _(YOLO):
    video_path = "input/HomeAloneTrailer.mp4"

    model = YOLO("yolov8n-oiv7.pt")
    TARGET_CLASS = "Spider"
    return TARGET_CLASS, model, video_path


@app.cell(hide_code=True)
def _(TARGET_CLASS, cv2, model, video_path):
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps  # Total video duration in seconds

    # Stores timestamps where spiders are detected
    spider_timestamps = []

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if the video ends
        
        results = model(frame, conf=0.001)

        # Check if spiders are detected
        detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
        if TARGET_CLASS in detected_classes:
            timestamp = frame_number / fps  # Convert frame number to seconds
            spider_timestamps.append(timestamp)

        frame_number += 1

    cap.release()
    return (
        cap,
        detected_classes,
        duration,
        fps,
        frame,
        frame_count,
        frame_number,
        results,
        ret,
        spider_timestamps,
        timestamp,
    )


@app.cell
def _(duration, plt, spider_timestamps):
    # Create timeline plot
    plt.figure(figsize=(10, 1))
    plt.bar(spider_timestamps, height=1, width=0.5, color='red', label="Spider Detected")
    plt.xlim(0, duration)
    plt.xlabel("Time (seconds)")
    plt.title("Phobia Detection Timeline")
    plt.legend()
    plt.yticks([])
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Show the graph
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
