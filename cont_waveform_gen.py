import sys
import os
import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ni_daq import NIDAQ


pp = pprint.PrettyPrinter(indent=4)
with NIDAQ() as daq:
    # Add both output channels
    daq.add_voltage_channel(["ao0", "ao1"], channel_type="ao")
    # Define waveform parameters
    waveforms = [
        {
            "frequency": 15.0,
            "amplitude": 1.0,
            "offset": 0,
            "waveform_type": "sawtooth",
        },
        {
            "frequency": 5.0,
            "amplitude": 0.5,
            "offset": 3,
            "waveform_type": "square",
            "duty_cycle": 0.3,
        },
    ]

    # Start generation
    daq.start_continuous_generation(waveform_params_list=waveforms)
    # Run for some time...
    input("Press Enter to stop waveform generation...\n")

    # Stop the generation
    daq.stop_continuous_generation()
