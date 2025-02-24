import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from queue import Queue
import threading
import pprint
from ni_daq import NIDAQ


def create_plot(
    time,
    data_buffer,
    channel_list,
    x_label_str="Time (s)",
    y_label_str="Voltage (V)",
    title_str="Real-time Waveform",
):
    """Create and initialize the plotting window."""

    # Create a list to store the lines.
    line_list = []

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, ch in enumerate(channel_list):
        (line,) = ax.plot(time, data_buffer[i], label=ch)
        line_list.append(line)
    # Set the title, y-axis label, and x-axis label of the plot.
    if title_str:
        ax.set_title(title_str)
    if y_label_str:
        ax.set_ylabel(y_label_str)
    if x_label_str:
        ax.set_xlabel(x_label_str)
    ax.legend()
    ax.grid()
    return fig, ax, line_list


def main():
    # Configuration parameters
    SAMPLE_RATE = 1000  # Hz
    SAMPLES_PER_UPDATE = 100
    BUFFER_SIZE = 1000
    CHANNEL = ["ai0", "ai2"]
    NUM_CHANNELS = len(CHANNEL)
    MIN_VOLTAGE = -5
    MAX_VOLTAGE = 5

    # Initialize buffers and utilities
    time = np.arange(BUFFER_SIZE) * (1 / SAMPLE_RATE)
    data_buffer = np.zeros((len(CHANNEL), BUFFER_SIZE))
    data_queue = Queue()
    running = threading.Event()
    running.set()

    # Create the initial plot
    fig, ax, line_list = create_plot(time, data_buffer, CHANNEL)

    def update_plot(frame):
        """Animation function to update the plot."""
        nonlocal data_buffer
        try:
            while not data_queue.empty():
                new_data = data_queue.get_nowait()
                if isinstance(new_data, np.ndarray):
                    # Reshape the new data if it's flat
                    if new_data.ndim == 1:
                        new_data = new_data.reshape(-1, NUM_CHANNELS).T

                    # Roll and update each channel's buffer
                    for i in range(NUM_CHANNELS):
                        data_buffer[i] = np.roll(data_buffer[i], -len(new_data[i]))
                        data_buffer[i, -len(new_data[i]) :] = new_data[i]

            # For each line in the list of lines, update the data of the line.
            for i, line in enumerate(line_list):
                line.set_ydata(data_buffer[i])

            ax.relim()
            ax.autoscale_view(True, True, True)
            return (line_list,)
        except Exception as e:
            print(f"Plot update error: {e}")
            return line_list

    def process_data(data, total_samples):
        """Callback function for the DAQ to process acquired data."""
        try:
            if data and running.is_set():
                data_queue.put(np.array(data))
        except Exception as e:
            print(f"Data processing error: {e}")

    # Initialize the DAQ
    with NIDAQ() as daq:
        # Configure the channel
        daq.add_voltage_channel(
            CHANNEL,
            min_val=MIN_VOLTAGE,
            max_val=MAX_VOLTAGE,
        )
        pp.pprint(daq.channels)

        # Set up the animation
        ani = FuncAnimation(
            fig,
            update_plot,
            interval=50,
            cache_frame_data=False,
        )

        # Start continuous acquisition
        daq.start_continuous_acquisition(
            callback_function=process_data,
            sample_rate=SAMPLE_RATE,
            samples_per_callback=SAMPLES_PER_UPDATE,
        )

        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nStopping acquisition...")
        finally:
            running.clear()
            daq.stop_continuous_acquisition()
            plt.close("all")


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
    main()
