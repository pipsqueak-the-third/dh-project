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
    video_path = "input/WALL-ETrailer.webm"
    return (video_path,)


@app.cell(hide_code=True)
def _(av, cv2, np, tqdm, video_path):
    """
    Berechnet die durchschnittliche Helligkeit eines Frames nach der Formel der Luminanz und speichert die Werte in einer Liste.

    """

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

                # Mittlerer Farbwert für jeden Frame
                mean_r = np.mean(pixels[:, :, 0])
                mean_g = np.mean(pixels[:, :, 1])
                mean_b = np.mean(pixels[:, :, 2])

                # Luminanz pro Frame
                brightness_val.append((0.2126 * mean_r +
                                   0.7152 * mean_g + 0.0722 * mean_b) / 255.0)

            last_s = next_s
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


@app.cell
def _(fft, frame_rate, np):
    """
    Untersucht auf schnell aufeinanderfolgende Helligkeitsunterschiede und blinkende Lichter, die photosensitive Epilepsie auslösen könnten.
    
    Parameter:
        brightness_val: Liste der errechneten durchschnittlichen Helligkeit pro Frame
        window_size (int): Größe des Fensters, um schnelle Änderungen zu erkennen
        threshold_percent (int): Prozentangabe, welche die Extremität der zu erkennenden Helligkeitsunterschiede bestimmt
        min_duration (float): Minimale Länge eines risikoreichen Intervalls (in Sekunden)
    
    Returns:
        "rapid_change_intervals": Intervalle, in denen schnelle & aufeinanderfolgende Änderungen der Helligkeit auftreten
        "high_risk_frequencies" (bool): Result der Frequenzanalyse mittels FFT, ob risikoreiche Frequenz enthalten ist 
    """

    def detect_epilepsy_risk(brightness_val, frame_rate=frame_rate, window_size=12, threshold_percent=95, min_duration=0.5):

        brightness = np.array(brightness_val)
        brightness_diff = np.abs(np.diff(brightness))

        rolling_diff = np.convolve(brightness_diff, np.ones(window_size), mode='valid')
        threshold = np.percentile(rolling_diff, threshold_percent)
    
        # Frames mit schnell aufeinanderfolgenden Helligkeitsänderungen finden
        rapid_changes = np.where(rolling_diff > threshold)[0]
    
        # Gruppieren von aufeinanderfolgenden gefundenen Sequenzen
        min_duration_frames = int(frame_rate * min_duration)
        intervals = []
        if len(rapid_changes) > 0:
            start = rapid_changes[0]
            for j in range(1, len(rapid_changes)):
                if rapid_changes[j] - rapid_changes[j - 1] > min_duration_frames:
                    intervals.append((start, rapid_changes[j - 1]))
                    start = rapid_changes[j]
            intervals.append((start, rapid_changes[-1]))
    
        # Frequenzanalyse mittels Fouriertransformation
        fft_vals = np.abs(fft(brightness))
        freqs = np.fft.fftfreq(len(brightness), d=1/frame_rate)
        risky_freqs = (freqs >= 3) & (freqs <= 30)
        high_risk_frequencies = np.any(fft_vals[risky_freqs] > np.percentile(fft_vals, threshold_percent))
    
        return {
            "rapid_change_intervals": intervals,
            "high_risk_frequencies": high_risk_frequencies
        }
    return (detect_epilepsy_risk,)


@app.cell
def _(brightness_val, detect_epilepsy_risk):
    results = detect_epilepsy_risk(brightness_val)
    intervals = results["rapid_change_intervals"]
    frequencies = results["high_risk_frequencies"]
    return frequencies, intervals, results


@app.cell
def _(brightness_val, frame_rate, np):
    """
    Hilfsfunktion um Zeit zu formatieren
    """
    time_axis_minutes = np.arange(len(brightness_val)) / frame_rate / 60
    
    def format_time(x, pos):
        minutes = int(x)
        seconds = int((x - minutes) * 60)
        return f'{minutes:02d}:{seconds:02d}'
    return format_time, time_axis_minutes


@app.cell
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


    """Helligkeitsgraph"""

    plt.subplot(gs[0])
    plt.plot(time_axis_minutes, brightness_val, color='blue')
    plt.title('Average Brightness of Video Frames')
    plt.xlabel('Time')
    plt.ylabel('Brightness')
    
    # Filtert gefundene Intervalle in "high-risk" (rot) und "lower-risk" (orange) ein, abhängig von der Länge des Intervalls (2 Sekunden)
    for start, end in intervals:
        duration = (end - start) / frame_rate
        if duration >= 2:
            plt.axvspan(start / frame_rate / 60, end / frame_rate / 60, color='red', alpha=0.3)
        else:
            plt.axvspan(start / frame_rate / 60, end / frame_rate / 60, color='orange', alpha=0.3)
    
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.25))
    plt.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)


    """Barcode"""

    plt.subplot(gs[1])
    plt.imshow([brightness_val], aspect='auto', cmap='gray', extent=[0, len(brightness_val) / frame_rate / 60, 0, 1])
    plt.title('Barcode Representation of Brightness')
    plt.xlabel('Time')
    plt.yticks([])

    # Filtert gefundene Intervalle in "high-risk" (rot) und "lower-risk" (orange) ein, abhängig von der Länge des Intervalls (2 Sekunden)
    for start, end in intervals:
        duration = (end - start) / frame_rate
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
