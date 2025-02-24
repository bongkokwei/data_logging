import sys
import os
import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ni_daq import NIDAQ


pp = pprint.PrettyPrinter(indent=4)
with NIDAQ() as daq:
    # Add single channel
    daq.add_voltage_channel("ai0", min_val=-5, max_val=5)

    # Add multiple channels
    daq.add_voltage_channel(["ai4", "ai5"], min_val=0, max_val=10)

    # Change channel limits
    daq.set_channel_limits("ai0", max_val=5.0)

    # Read single sample from all channels
    data = daq.read_samples()
    pp.pprint(data)

    # Read multiple samples and reshape for multiple channels
    data = daq.read_samples(clk_rate=1000, num_samples=64)
    pp.pprint(data)

    # Get channel information
    info = daq.get_channel_info("ai0")
    pp.pprint(info)
    pp.pprint(daq.channels)
