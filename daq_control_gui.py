import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QLineEdit,
    QGridLayout,
    QGroupBox,
    QTextEdit,
    QFrame,
)
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt
import pyqtgraph as pg
from ni_daq import NIDAQ
import re


class DataAcquisitionWorker(QObject):
    """Worker class for handling continuous data acquisition"""

    data_ready = pyqtSignal(object)

    def __init__(self, sample_rate=1000, samples_per_update=100):
        super().__init__()
        self.sample_rate = sample_rate
        self.samples_per_update = samples_per_update
        self.running = False
        self.daq = None
        self.channels = []
        self.min_val = -5
        self.max_val = 5

    def process_data(self, data, total_samples):
        """Callback function for the DAQ to process acquired data"""
        if data and self.running:
            self.data_ready.emit(np.array(data))

    def start_acquisition(self):
        """Start the data acquisition process"""
        if not self.running and self.channels:
            self.daq = NIDAQ()
            self.daq.__enter__()
            self.daq.add_voltage_channel(
                self.channels,
                min_val=self.min_val,
                max_val=self.max_val,
                channel_type="ai",
            )
            self.running = True
            self.daq.start_continuous_acquisition(
                callback_function=self.process_data,
                sample_rate=self.sample_rate,
                samples_per_callback=self.samples_per_update,
            )

    def stop_acquisition(self):
        """Stop the data acquisition process"""
        if self.running:
            self.running = False
            self.daq.stop_continuous_acquisition()
            self.daq.__exit__(None, None, None)
            self.daq = None


class WaveformGenerationWorker(QObject):
    """Worker class for handling waveform generation"""

    def __init__(self):
        super().__init__()
        self.running = False
        self.daq = None
        self.channels = []
        self.waveform_params = []

    def start_generation(self):
        """Start the waveform generation process"""
        if not self.running and self.channels and self.waveform_params:
            self.daq = NIDAQ()
            self.daq.__enter__()
            self.daq.add_voltage_channel(self.channels, channel_type="ao")
            self.running = True
            self.daq.start_continuous_generation(
                waveform_params_list=self.waveform_params
            )

    def stop_generation(self):
        """Stop the waveform generation process"""
        if self.running:
            self.running = False
            self.daq.stop_continuous_generation()
            self.daq.__exit__(None, None, None)
            self.daq = None


class DAQControlGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAQ Control Interface")
        self.setGeometry(100, 100, 1200, 800)

        self.buffer_size = 1000
        self.sample_rate = 1000  # Hz
        self.samples_per_update = 100
        self.window_size = 50

        # Initialize workers and threads
        self.init_workers()

        # Initialize channel parameters
        self.ai_params = {}  # Store input channel parameters
        self.channel_colors = {}  # Store channel colors

        # Initialize time and data buffers
        self.time_data = (
            np.arange(self.buffer_size) / self.sample_rate
        )  # Time in seconds
        self.num_channels = 0
        self.data_buffer = None

        # Create the main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QGridLayout(main_widget)

        # Create control panel and plot area
        control_panel = self.create_control_panel()
        plot_area = self.create_plot_area()
        display_panel = self.create_display_panel()

        # Add widgets to grid layout
        layout.addWidget(control_panel, 0, 0, 2, 1)
        layout.addWidget(plot_area, 0, 1)
        layout.addWidget(display_panel, 1, 1)

        # Set column stretch factors
        layout.setColumnStretch(0, 1)  # Control panel column
        layout.setColumnStretch(1, 8)  # Plot area column

        # Leave row 1 empty for future widgets
        layout.setRowStretch(0, 8)
        layout.setRowStretch(1, 1)

        # Initialize plot data
        self.num_channels = 0
        self.data_buffer = None
        self.init_plot_data()

    def init_workers(self):
        """Initialize worker threads for acquisition and generation"""
        # Acquisition worker
        self.acq_thread = QThread()
        self.acq_worker = DataAcquisitionWorker(
            sample_rate=self.sample_rate,
            samples_per_update=self.samples_per_update,
        )
        self.acq_worker.moveToThread(self.acq_thread)
        self.acq_worker.data_ready.connect(self.update_plot)
        self.acq_worker.data_ready.connect(self.update_display)
        self.acq_thread.start()

        # Generation worker
        self.gen_thread = QThread()
        self.gen_worker = WaveformGenerationWorker()
        self.gen_worker.moveToThread(self.gen_thread)
        self.gen_thread.start()

    def create_display_panel(self):
        display_panel = QGroupBox("Display Panel")
        display_layout = QGridLayout()
        num_input_channel = 8
        num_row = 2
        num_col = np.ceil(num_input_channel / num_row)
        curr_row = curr_col = 0
        self.ai_output_text = {}

        for chn in range(num_input_channel):
            text_box = QTextEdit()
            text_box.setReadOnly(True)
            text_box.setText(f"ai{chn}: {0.0:.4f} V")
            text_box.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            text_box.setMinimumSize(100, 50)  # Adjust size as needed
            font = text_box.font()
            font.setPointSize(14)  # Adjust size as needed
            text_box.setFont(font)
            text_box.setFrameStyle(QFrame.Panel | QFrame.Sunken)
            text_box.setStyleSheet("background-color: lightblue;")

            # Store in dictionary with key "ai{chn}"
            self.ai_output_text[f"ai{chn}"] = text_box

            display_layout.addWidget(text_box, curr_row, curr_col)
            if curr_col < num_col - 1:
                curr_col += 1
            else:
                curr_row += 1
                curr_col = 0

        display_panel.setLayout(display_layout)
        return display_panel

    def create_control_panel(self):
        """Create the control panel with all input widgets"""
        control_panel = QGroupBox("Control Panel")
        layout = QVBoxLayout()

        # Analog Input Controls
        ai_group = QGroupBox("Analog Input")
        ai_layout = QGridLayout()

        self.ai_channel_combo = QComboBox()
        self.ai_channel_combo.addItems([f"ai{i}" for i in range(8)])
        self.ai_channels_selected = []

        self.min_val_input = QLineEdit("-5")
        self.max_val_input = QLineEdit("5")

        self.add_ai_button = QPushButton("Add Channel")
        self.add_ai_button.clicked.connect(self.add_ai_channel)

        self.start_ai_button = QPushButton("Start Acquisition")
        self.start_ai_button.clicked.connect(self.start_acquisition)

        self.stop_ai_button = QPushButton("Stop Acquisition")
        self.stop_ai_button.clicked.connect(self.stop_acquisition)

        ai_layout.addWidget(QLabel("Channel:"), 0, 0)
        ai_layout.addWidget(self.ai_channel_combo, 0, 1)
        ai_layout.addWidget(QLabel("Min Value:"), 1, 0)
        ai_layout.addWidget(self.min_val_input, 1, 1)
        ai_layout.addWidget(QLabel("Max Value:"), 2, 0)
        ai_layout.addWidget(self.max_val_input, 2, 1)
        ai_layout.addWidget(self.add_ai_button, 3, 0)
        ai_layout.addWidget(self.start_ai_button, 4, 0)
        ai_layout.addWidget(self.stop_ai_button, 4, 1)

        ai_group.setLayout(ai_layout)

        # Analog Output Controls
        ao_group = QGroupBox("Analog Output")
        ao_layout = QGridLayout()

        self.ao_channel_combo = QComboBox()
        self.ao_channel_combo.addItems([f"ao{i}" for i in range(2)])

        self.waveform_type_combo = QComboBox()
        self.waveform_type_combo.addItems(["sine", "square", "sawtooth", "triangle"])

        self.frequency_input = QLineEdit("10.0")
        self.amplitude_input = QLineEdit("1.0")
        self.offset_input = QLineEdit("0.0")
        self.duty_cycle_input = QLineEdit("0.5")

        self.add_ao_button = QPushButton("Add Output")
        self.add_ao_button.clicked.connect(self.add_ao_channel)

        self.start_ao_button = QPushButton("Start Generation")
        self.start_ao_button.clicked.connect(self.start_generation)

        self.stop_ao_button = QPushButton("Stop Generation")
        self.stop_ao_button.clicked.connect(self.stop_generation)

        ao_layout.addWidget(QLabel("Channel:"), 0, 0)
        ao_layout.addWidget(self.ao_channel_combo, 0, 1)
        ao_layout.addWidget(QLabel("Waveform:"), 1, 0)
        ao_layout.addWidget(self.waveform_type_combo, 1, 1)
        ao_layout.addWidget(QLabel("Frequency (Hz):"), 2, 0)
        ao_layout.addWidget(self.frequency_input, 2, 1)
        ao_layout.addWidget(QLabel("Amplitude:"), 3, 0)
        ao_layout.addWidget(self.amplitude_input, 3, 1)
        ao_layout.addWidget(QLabel("Offset:"), 4, 0)
        ao_layout.addWidget(self.offset_input, 4, 1)
        ao_layout.addWidget(QLabel("Duty Cycle:"), 5, 0)
        ao_layout.addWidget(self.duty_cycle_input, 5, 1)
        ao_layout.addWidget(self.add_ao_button, 6, 0)
        ao_layout.addWidget(self.start_ao_button, 7, 0)
        ao_layout.addWidget(self.stop_ao_button, 7, 1)

        ao_group.setLayout(ao_layout)

        # Create channel display and parameters group
        channel_display_group = QGroupBox("Active Channels and Parameters")
        channel_display_layout = QGridLayout()

        # Input channels section
        input_group = QGroupBox("Input Channels")
        input_layout = QVBoxLayout()

        # Input channel selection and removal
        input_control_layout = QHBoxLayout()
        self.ai_channels_display = QComboBox()
        self.ai_channels_display.setPlaceholderText("No input channels added")
        self.ai_channels_display.currentTextChanged.connect(
            self.update_ai_params_display
        )
        input_control_layout.addWidget(QLabel("Channel:"))
        input_control_layout.addWidget(self.ai_channels_display)

        self.remove_ai_button = QPushButton("Remove Channel")
        self.remove_ai_button.clicked.connect(self.remove_ai_channel)
        input_control_layout.addWidget(self.remove_ai_button)
        input_layout.addLayout(input_control_layout)

        # Input parameters display
        self.ai_params_text = QTextEdit()
        self.ai_params_text.setReadOnly(True)
        self.ai_params_text.setMaximumHeight(100)
        input_layout.addWidget(self.ai_params_text)

        input_group.setLayout(input_layout)

        # Output channels section
        output_group = QGroupBox("Output Channels")
        output_layout = QVBoxLayout()

        # Output channel selection and removal
        output_control_layout = QHBoxLayout()
        self.ao_channels_display = QComboBox()
        self.ao_channels_display.setPlaceholderText("No output channels added")
        self.ao_channels_display.currentTextChanged.connect(
            self.update_ao_params_display
        )
        output_control_layout.addWidget(QLabel("Channel:"))
        output_control_layout.addWidget(self.ao_channels_display)

        self.remove_ao_button = QPushButton("Remove Channel")
        self.remove_ao_button.clicked.connect(self.remove_ao_channel)
        output_control_layout.addWidget(self.remove_ao_button)
        output_layout.addLayout(output_control_layout)

        # Output parameters display
        self.ao_params_text = QTextEdit()
        self.ao_params_text.setReadOnly(True)
        self.ao_params_text.setMaximumHeight(100)
        output_layout.addWidget(self.ao_params_text)

        output_group.setLayout(output_layout)

        # Add groups to main layout
        channel_display_layout.addWidget(input_group, 0, 0)
        channel_display_layout.addWidget(output_group, 1, 0)
        channel_display_group.setLayout(channel_display_layout)

        # Add groups to control panel
        layout.addWidget(ai_group)
        layout.addWidget(ao_group)
        layout.addWidget(channel_display_group)
        layout.addStretch()

        control_panel.setLayout(layout)
        return control_panel

    def create_plot_area(self):
        """Create the plotting area"""
        plot_group = QGroupBox("Real-time Plot")
        layout = QVBoxLayout()

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel("left", "Voltage (V)")
        self.plot_widget.setLabel("bottom", "Time (s)")

        layout.addWidget(self.plot_widget)
        plot_group.setLayout(layout)

        return plot_group

    def init_plot_data(self):
        """Initialize the plot data buffers"""
        self.plot_curves = []
        self.data_buffer = None
        self.plot_widget.clear()

    def add_ai_channel(self):
        """Add an analog input channel to the acquisition list"""
        channel = self.ai_channel_combo.currentText()
        if channel not in self.ai_channels_selected:
            self.ai_channels_selected.append(channel)
            self.num_channels = len(self.ai_channels_selected)
            self.data_buffer = np.zeros((self.num_channels, self.buffer_size))

            # Generate and store color for this channel
            colors = [
                (76, 114, 176),  # Blue
                (221, 132, 82),  # Orange
                (85, 168, 104),  # Green
                (196, 78, 82),  # Red
                (129, 114, 178),  # Purple
                (147, 120, 96),  # Brown
                (218, 139, 195),  # Pink
                (140, 140, 140),  # Gray
            ]

            color_idx = len(self.plot_curves) % len(colors)
            color = colors[color_idx]
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            self.channel_colors[channel] = color_hex

            # Add new plot curve with color and name
            curve = self.plot_widget.plot(
                pen=pg.mkPen(color=color, width=2), name=channel
            )
            self.plot_curves.append(curve)

            # Store channel parameters
            self.ai_params[channel] = {
                "min_val": float(self.min_val_input.text()),
                "max_val": float(self.max_val_input.text()),
                "color": color_hex,
            }

            # Update channels display
            self.ai_channels_display.clear()
            self.ai_channels_display.addItems(self.ai_channels_selected)
            self.update_ai_params_display(channel)

    def update_ai_params_display(self, channel):
        """Update the display of input channel parameters"""
        if channel in self.ai_params:
            params = self.ai_params[channel]
            chn_disp = re.sub(r"ai(\d+)", r"Analog input \1", channel)
            color_style = f"color: {params['color']}; font-weight: bold;"
            html_text = f"""
                <div style='font-family: monospace;'>
                <p style='margin:0; {color_style}'>{chn_disp}</p>
                <p style='margin:0; {color_style}'>Min Value: {params['min_val']} V</p>
                <p style='margin:0; {color_style}'>Max Value: {params['max_val']} V</p>
                </div>
            """
            self.ai_params_text.setHtml(html_text)
        else:
            self.ai_params_text.clear()

    def add_ao_channel(self):
        """Add an analog output channel with waveform parameters"""
        channel = self.ao_channel_combo.currentText()
        if channel not in self.gen_worker.channels:
            self.gen_worker.channels.append(channel)

            # Create and store waveform parameters
            params = {
                "frequency": float(self.frequency_input.text()),
                "amplitude": float(self.amplitude_input.text()),
                "offset": float(self.offset_input.text()),
                "waveform_type": self.waveform_type_combo.currentText(),
                "duty_cycle": float(self.duty_cycle_input.text()),
            }
            self.gen_worker.waveform_params.append(params)

            # Update channels display
            self.ao_channels_display.clear()
            self.ao_channels_display.addItems(self.gen_worker.channels)
            self.update_ao_params_display(channel)

    def update_ao_params_display(self, channel):
        """Update the display of output channel parameters"""
        if channel in self.gen_worker.channels:
            idx = self.gen_worker.channels.index(channel)
            chn_disp = re.sub(r"ao(\d+)", r"Analog output \1", channel)
            if idx < len(self.gen_worker.waveform_params):
                params = self.gen_worker.waveform_params[idx]
                html_text = f"""
                    <div style='font-family: monospace;'>
                    <p style='font-weight: bold; margin:0'>{chn_disp}</p>
                    <p style='margin: 0'>Type: {params['waveform_type']}</p>
                    <p style='margin: 0'>Frequency: {params['frequency']} Hz</p>
                    <p style='margin: 0'>Amplitude: {params['amplitude']} V</p>
                    <p style='margin: 0'>Offset: {params['offset']} V</p>
                    <p style='margin: 0'>Duty Cycle: {params['duty_cycle']}</p>
                    </div>
                """
                self.ao_params_text.setHtml(html_text)
        else:
            self.ao_params_text.clear()

    def remove_ai_channel(self):
        """Remove selected analog input channel"""
        if not self.ai_channels_selected:
            return

        channel = self.ai_channels_display.currentText()
        if channel in self.ai_channels_selected:
            # Get index of channel to remove
            idx = self.ai_channels_selected.index(channel)

            # Remove channel from list and update display
            self.ai_channels_selected.remove(channel)
            self.ai_channels_display.clear()
            self.ai_channels_display.addItems(self.ai_channels_selected)

            # Remove corresponding plot curve and parameters
            if idx < len(self.plot_curves):
                self.plot_widget.removeItem(self.plot_curves[idx])
                self.plot_curves.pop(idx)

            # Remove channel parameters
            if channel in self.ai_params:
                del self.ai_params[channel]
            if channel in self.channel_colors:
                del self.channel_colors[channel]

            # Update buffer size
            self.num_channels = len(self.ai_channels_selected)
            if self.num_channels > 0:
                self.data_buffer = np.zeros((self.num_channels, self.buffer_size))
            else:
                self.data_buffer = None

            # Clear parameters display if no channel selected
            if not self.ai_channels_selected:
                self.ai_params_text.clear()

    def add_ao_channel(self):
        """Add an analog output channel with waveform parameters"""
        channel = self.ao_channel_combo.currentText()
        if channel not in self.gen_worker.channels:
            self.gen_worker.channels.append(channel)

            # Create waveform parameters
            params = {
                "frequency": float(self.frequency_input.text()),
                "amplitude": float(self.amplitude_input.text()),
                "offset": float(self.offset_input.text()),
                "waveform_type": self.waveform_type_combo.currentText(),
                "duty_cycle": float(self.duty_cycle_input.text()),
            }
            self.gen_worker.waveform_params.append(params)

            # Update channels display
            self.ao_channels_display.clear()
            self.ao_channels_display.addItems(self.gen_worker.channels)
            self.update_ao_params_display(channel)

    def remove_ao_channel(self):
        """Remove selected analog output channel"""
        if not self.gen_worker.channels:
            return

        channel = self.ao_channels_display.currentText()
        if channel in self.gen_worker.channels:
            # Get index of channel to remove
            idx = self.gen_worker.channels.index(channel)

            # Remove channel and its parameters
            self.gen_worker.channels.remove(channel)
            if idx < len(self.gen_worker.waveform_params):
                self.gen_worker.waveform_params.pop(idx)

            # Update display
            self.ao_channels_display.clear()
            self.ao_channels_display.addItems(self.gen_worker.channels)

    def start_acquisition(self):
        """Start the data acquisition"""
        if self.ai_channels_selected:
            self.acq_worker.channels = self.ai_channels_selected
            self.acq_worker.min_val = float(self.min_val_input.text())
            self.acq_worker.max_val = float(self.max_val_input.text())
            self.acq_worker.start_acquisition()

    def stop_acquisition(self):
        """Stop the data acquisition"""
        self.acq_worker.stop_acquisition()

    def start_generation(self):
        """Start the waveform generation"""
        self.gen_worker.start_generation()

    def stop_generation(self):
        """Stop the waveform generation"""
        self.gen_worker.stop_generation()

    def update_plot(self, new_data):
        """Update the plot with new data"""
        if isinstance(new_data, np.ndarray):
            if new_data.ndim == 1:
                new_data = new_data.reshape(-1, self.num_channels).T

            for i in range(self.num_channels):
                if i < len(self.plot_curves):
                    self.data_buffer[i] = np.roll(
                        self.data_buffer[i], -len(new_data[i])
                    )
                    self.data_buffer[i, -len(new_data[i]) :] = new_data[i]
                    self.plot_curves[i].setData(self.time_data, self.data_buffer[i])

    def update_display(self, new_data):
        if isinstance(new_data, np.ndarray):
            if new_data.ndim == 1:
                new_data = new_data.reshape(-1, self.num_channels).T
        new_data_mean = np.mean(new_data[:, : self.window_size], axis=1)

        for ii, chn in enumerate(self.ai_channels_selected):
            text_box = self.ai_output_text[chn]
            text_box.setText(f"{chn}: {new_data_mean[ii]:.4f} V")
            text_box.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

    def closeEvent(self, event):
        """Clean up when closing the application"""
        self.stop_acquisition()
        self.stop_generation()
        self.acq_thread.quit()
        self.gen_thread.quit()
        self.acq_thread.wait()
        self.gen_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DAQControlGUI()
    window.show()
    sys.exit(app.exec_())
