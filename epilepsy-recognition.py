import marimo

__generated_with = "0.11.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import av
    import sys
    import cv2
    import PIL.Image
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.fftpack import fft
    from tqdm import tqdm
    return PIL, av, cv2, fft, gridspec, mo, np, plt, sys, tqdm


@app.cell
def _():
    video_path = "input/WildAtHeartTrailer.mp4"
    return (video_path,)


@app.cell
def _(av, cv2, np, tqdm, video_path):
    brightness_val = []

    with av.open(video_path) as f:
        last_s = -1
        stream = f.streams.video[0]
        n_frames = stream.frames
        cap = cv2.VideoCapture(video_path)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        for i, frame in enumerate(tqdm(f.decode(stream), total=n_frames)):
            next_s = int(frame.time)
            if next_s >= last_s:
                pixels = frame.to_ndarray(format="rgb24")

                # Mean color for each frame
                mean_r = np.mean(pixels[:, :, 0])
                mean_g = np.mean(pixels[:, :, 1])
                mean_b = np.mean(pixels[:, :, 2])

                brightness_val.append((0.2126 * mean_r +
                                   0.7152 * mean_g + 0.0722 * mean_b) / 255.0)

            last_s = next_s

    print(brightness_val)
    return (
        brightness_val,
        cap,
        f,
        frame,
        frame_rate,
        i,
        last_s,
        mean_b,
        mean_g,
        mean_r,
        n_frames,
        next_s,
        pixels,
        stream,
    )


@app.cell(hide_code=True)
def _(fft, frame_rate, np):
    def detect_epilepsy_risk(brightness_val, frame_rate=frame_rate, window_size=12, threshold_percentile=95, min_flash_duration=0.5):
        """
        Detects high-risk rapid brightness changes and flashing sequences that could trigger epilepsy.
    
        Parameters:
            brightness_vals (list or np.array): List of brightness values over time.
            frame_rate (int): Frames per second of the video (default 60 FPS).
            window_size (int): Size of the rolling window for detecting rapid changes.
            threshold_percentile (int): Percentile for defining extreme changes.
            min_flash_duration (float): Minimum duration (in seconds) for high-risk intervals.
    
        Returns:
            dict: Contains rapid change intervals and frequency analysis results.
        """
        brightness = np.array(brightness_val)
        brightness_diff = np.abs(np.diff(brightness))
    
        # Apply a rolling sum of brightness differences
        rolling_diff = np.convolve(brightness_diff, np.ones(window_size), mode='valid')
        threshold = np.percentile(rolling_diff, threshold_percentile)
    
        # Find frames with rapid brightness changes
        rapid_changes = np.where(rolling_diff > threshold)[0]
    
        # Group consecutive rapid changes
        min_duration_frames = int(frame_rate * min_flash_duration)
        intervals = []
        if len(rapid_changes) > 0:
            start = rapid_changes[0]
            for j in range(1, len(rapid_changes)):
                if rapid_changes[j] - rapid_changes[j - 1] > min_duration_frames:
                    intervals.append((start, rapid_changes[j - 1]))
                    start = rapid_changes[j]
            intervals.append((start, rapid_changes[-1]))
    
        # Frequency analysis using FFT
        fft_vals = np.abs(fft(brightness))
        freqs = np.fft.fftfreq(len(brightness), d=1/frame_rate)
        risky_freqs = (freqs >= 3) & (freqs <= 30)
        high_risk_frequencies = np.any(fft_vals[risky_freqs] > np.percentile(fft_vals, threshold_percentile))

        print(intervals)
        print(high_risk_frequencies)
    
        return {
            "rapid_change_intervals": intervals,
            "high_risk_frequencies_detected": high_risk_frequencies
        }
    return (detect_epilepsy_risk,)


@app.cell(hide_code=True)
def _(brightness_val, detect_epilepsy_risk):
    results = detect_epilepsy_risk(brightness_val)
    intervals = results["rapid_change_intervals"]
    frequencies = results["high_risk_frequencies_detected"]
    return frequencies, intervals, results


@app.cell(hide_code=True)
def _(brightness_val, frame_rate, np):
    # format time in mm:ss
    time_axis_minutes = np.arange(len(brightness_val)) / frame_rate / 60
    
    def format_time(x, pos):
        minutes = int(x)
        seconds = int((x - minutes) * 60)
        return f'{minutes:02d}:{seconds:02d}'
    return format_time, time_axis_minutes


@app.cell(hide_code=True)
def _(
    brightness_val,
    format_time,
    frame_rate,
    frequencies,
    gridspec,
    intervals,
    plt,
    time_axis_minutes,
):
    plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.85, 0.15])
    
    # Brightness plot
    plt.subplot(gs[0])
    plt.plot(time_axis_minutes, brightness_val, color='blue')
    plt.title('Average Brightness of Video Frames')
    plt.xlabel('Time')
    plt.ylabel('Brightness')
    
    # Filtered trigger intervals
    for start, end in intervals:
        duration = (end - start) / frame_rate  # in seconds
        if duration >= 2:  # 2 seconds threshold
            plt.axvspan(start / frame_rate / 60, end / frame_rate / 60, color='red', alpha=0.3)
        else:
            plt.axvspan(start / frame_rate / 60, end / frame_rate / 60, color='orange', alpha=0.3)
    
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.25))  # major ticks every 15 seconds
    plt.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
    
    # Barcode plot
    plt.subplot(gs[1])
    plt.imshow([brightness_val], aspect='auto', cmap='gray', extent=[0, len(brightness_val) / frame_rate / 60, 0, 1])
    plt.title('Barcode Representation of Brightness')
    plt.xlabel('Time')
    plt.yticks([])
    
    for start, end in intervals:
        duration = (end - start) / frame_rate  # in seconds
        if duration >= 2:
            plt.axvspan(start / frame_rate / 60, end / frame_rate / 60, color='red', alpha=0.5)
        else:
            plt.axvspan(start / frame_rate / 60, end / frame_rate / 60, color='orange', alpha=0.5)
    
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.25))

    if frequencies:
        plt.figtext(0.5, -0.1, "High risk frequencies detected.", ha="center", fontsize=12)
    else:
        plt.figtext(0.5, -0.1, "No high risk frequencies detected.", ha="center", fontsize=12)

    plt.tight_layout()
    plt.show()



    return duration, end, gs, start


if __name__ == "__main__":
    app.run()
