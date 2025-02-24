import sys
import os
import pprint
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ni_daq import NIDAQ

pp = pprint.PrettyPrinter(indent=4)


# Example of continuous acquisition:
def process_data(data, total_samples):
    """Example callback function to process acquired data."""
    print(
        f"Acquired {len(data)} samples. Total: {total_samples} | "
        f"Data: {np.array2string(np.array(data)[:5], precision=4)}...",
        end="\r",
    )


with NIDAQ() as daq:
    # Add channel for continuous acquisition
    daq.add_voltage_channel("ai0", min_val=-5, max_val=5)

    # Start continuous acquisition
    daq.start_continuous_acquisition(
        callback_function=process_data,
        sample_rate=100.0,
        samples_per_callback=100,
    )

    input("Press Enter to stop continuous acquisition...\n")

    # Stop acquisition
    daq.stop_continuous_acquisition()
    print(f"\nTotal samples acquired: {daq.total_samples_read}")
