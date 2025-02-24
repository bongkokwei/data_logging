# DAQ Control GUI

A Python-based graphical user interface for controlling National Instruments Data Acquisition (NI-DAQ) devices. This application provides real-time data visualisation and waveform generation capabilities.

## Features

- **Real-time Data Acquisition**
  - Multiple analog input channel support
  - Configurable voltage ranges
  - Real-time plotting with time-based x-axis
  - Color-coded channels for easy identification

- **Waveform Generation**
  - Multiple analog output channel support
  - Multiple waveform types (sine, square, sawtooth, triangle)
  - Configurable parameters (frequency, amplitude, offset, duty cycle)

- **User Interface**
  - Intuitive channel management
  - Parameter display for active channels
  - Real-time data visualisation
  - Separate threads for acquisition and generation

## Prerequisites

- Python 3.x
- NI-DAQ drivers installed

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd daq-control-gui
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python daq_control_gui.py
   ```

2. Adding Input Channels:
   - Select analog input channel from dropdown
   - Set minimum and maximum voltage values
   - Click "Add Channel"
   - View real-time data in the plot

3. Generating Waveforms:
   - Select analog output channel
   - Choose waveform type
   - Set parameters (frequency, amplitude, etc.)
   - Click "Add Output"
   - Start/stop generation as needed