import marimo

__generated_with = "0.11.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import av
    import sys
    import PIL.Image
    import numpy as np
    import matplotlib.pyplot as plt

    from tqdm import tqdm
    return PIL, av, mo, np, plt, sys, tqdm


@app.cell
def _():
    video_path = "input/output_24fps.webm"
    return (video_path,)


@app.cell(hide_code=True)
def _(av, np, tqdm, video_path):
    brightness_val = []

    with av.open(video_path) as f:
        last_s = -1
        stream = f.streams.video[0]
        n_frames = stream.frames
        frame_rate = stream.average_rate if stream.average_rate else 30  # Fallback if unknown
        for i, frame in enumerate(tqdm(f.decode(stream), total=n_frames)):
            next_s = int(frame.time)
            if next_s >= last_s:
                pixels = frame.to_ndarray(format="rgb24")

                # Calculate the mean color for each frame
                mean_r = np.mean(pixels[:, :, 0])
                mean_g = np.mean(pixels[:, :, 1])
                mean_b = np.mean(pixels[:, :, 2])

                # Calculate brightness using the standard luminance formula
                brightness_val.append((0.2126 * mean_r +
                                   0.7152 * mean_g + 0.0722 * mean_b) / 255.0)

            last_s = next_s
    return (
        brightness_val,
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
def _(brightness_val, np):
    brightness = np.array(brightness_val)

    brightness_diff = np.abs(np.diff(brightness))

    threshold = np.percentile(brightness_diff, 85)

    rapid_changes = np.where(brightness_diff > threshold)[0]

    if len(rapid_changes) > 0:
        intervals = []
        start = rapid_changes[0]
        for j in range(1, len(rapid_changes)):
            if rapid_changes[j] != rapid_changes[j - 1] + 1:
                intervals.append((start, rapid_changes[j - 1]))  # Store interval
                start = rapid_changes[j]
        intervals.append((start, rapid_changes[-1]))  # Add last interval
    else:
        print("No significant rapid brightness changes detected.")
    return (
        brightness,
        brightness_diff,
        intervals,
        j,
        rapid_changes,
        start,
        threshold,
    )


@app.cell(hide_code=True)
def _(intervals):
    def process_intervals(intervals, merge_threshold=12, min_length=12):
        merged_intervals = []
    
        if intervals:
            start, end = intervals[0]

            for i in range(1, len(intervals)):
                next_start, next_end = intervals[i]

                if next_start - end <= merge_threshold:
                    end = next_end
                else:
                    merged_intervals.append((start, end))
                    start, end = next_start, next_end

            merged_intervals.append((start, end))

        # Remove short intervals
        filtered_intervals = [(s, e) for s, e in merged_intervals if (e - s) >= min_length]
    
        return filtered_intervals

    # Call the function and store the result globally
    filtered_intervals = process_intervals(intervals)
    return filtered_intervals, process_intervals


@app.cell(hide_code=True)
def _(brightness_val, filtered_intervals, np, plt):
    def _():
        # Convert frame index to time in minutes
        frame_rate = 24  # Video frame rate (frames per second)
        time_axis_minutes = np.arange(len(brightness_val)) / frame_rate / 60  # Time in minutes for each frame
    
        # Helper function to format time in mm:ss
        def format_time(x, pos):
            minutes = int(x)
            seconds = int((x - minutes) * 60)
            return f'{minutes:02d}:{seconds:02d}'
    
        # Create the figure and axis for the plots
        plt.figure(figsize=(10, 6))
    
        # Brightness plot with video-like timeline
        plt.subplot(2, 1, 1)
        plt.plot(time_axis_minutes, brightness_val, color='blue')
        plt.title('Average Brightness of Video Frames')
        plt.xlabel('Time')  # Label the x-axis with video time format
        plt.ylabel('Brightness')
    
        # Highlight filtered trigger intervals (convert start and end times to minutes)
        for start, end in filtered_intervals:
            plt.axvspan(start / frame_rate / 60, end / frame_rate / 60, color='red', alpha=0.3)
    
        # Format the x-axis to display time in mm:ss format
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    
        # Set the major and minor ticks for the x-axis (every 30 seconds and finer detail)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.25))  # Major ticks every 15 seconds

    
        # Add a grid to the x-axis for better readability
        plt.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
    
        # Barcode-like plot with video timeline on the x-axis
        plt.subplot(2, 1, 2)
        plt.imshow([brightness_val], aspect='auto', cmap='gray', extent=[0, len(brightness_val) / frame_rate / 60, 0, 1])
        plt.title('Barcode Representation of Brightness')
        plt.xlabel('Time')  # Label the x-axis for the barcode plot
        plt.yticks([])  # Remove y-axis ticks for the barcode plot
    
        # Highlight filtered trigger intervals (convert start and end times to minutes)
        for start, end in filtered_intervals:
            plt.axvspan(start / frame_rate / 60, end / frame_rate / 60, color='red', alpha=0.5)
    
        # Format the x-axis to display time in mm:ss for the barcode-like plot
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_time))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.25))  # Major ticks every 15 seconds
    
        # Adjust layout to prevent overlap
        plt.tight_layout()
    
        # Display the plots
        plt.show()


    _()
    return


if __name__ == "__main__":
    app.run()
