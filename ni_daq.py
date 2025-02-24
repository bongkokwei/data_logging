import nidaqmx
import numpy as np
import pprint
from scipy import signal
from typing import (
    Union,
    List,
    Optional,
    Callable,
    Tuple,
)
from nidaqmx.constants import (
    READ_ALL_AVAILABLE,
    AcquisitionType,
    TerminalConfiguration,
)


class NIDAQ:
    """
    A class to handle National Instruments DAQ operations.
    (https://github.com/ni/nidaqmx-python)
    Provides interface for channel configuration and data acquisition.
    """

    def __init__(self, device_name: str = "Dev1"):
        """
        Initialize DAQ with device name.

        Args:
            device_name (str): Name of the DAQ device (default: "Dev1")
        """
        self.device_name = device_name
        self.task = None
        self.channels = {}
        self._continuous_acquisition_running = False
        self._continuous_generation_running = False
        self.ao_task = None

    def __enter__(self):
        """Context manager entry point."""
        self.task = nidaqmx.Task()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        if self.task:
            self.task.close()

    def add_voltage_channel(
        self,
        channel: Union[str, List[str]],
        min_val: float = -10.0,
        max_val: float = 10.0,
        channel_type: str = "ai",
    ) -> None:
        """
        Add analog voltage channel(s) for input or output.

        Args:
            channel: Channel identifier (e.g., "ai0" or "ao0")
            min_val: Minimum voltage value
            max_val: Maximum voltage value
            channel_type: Type of channel ("ai" for input or "ao" for output)
        """
        if isinstance(channel, str):
            channel_name = [f"{self.device_name}/{channel}"]
        else:
            channel_name = [f"{self.device_name}/{ch}" for ch in channel]

        for ch in channel_name:
            if channel_type == "ai":
                channel_obj = self.task.ai_channels.add_ai_voltage_chan(
                    ch,
                    min_val=min_val,
                    max_val=max_val,
                    terminal_config=TerminalConfiguration.DEFAULT,
                )
            elif channel_type == "ao":
                channel_obj = self.task.ao_channels.add_ao_voltage_chan(
                    ch,
                    min_val=min_val,
                    max_val=max_val,
                )
            else:
                raise ValueError(f"Unsupported channel type: {channel_type}")

            self.channels[ch] = channel_obj

    def set_channel_limits(
        self,
        channel: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> None:
        """
        Set voltage limits for a specific channel.

        Args:
            channel: Channel identifier
            min_val: New minimum voltage value
            max_val: New maximum voltage value
        """
        channel_name = f"{self.device_name}/{channel}"
        if channel_name in self.channels:
            if min_val is not None:
                self.channels[channel_name].ai_min = min_val
            if max_val is not None:
                self.channels[channel_name].ai_max = max_val

    def read_samples(
        self,
        num_samples: int = 2,  # minimum number of samples
        clk_rate: float = 1000,
        reshape_output: bool = False,
    ) -> np.ndarray:
        """
        Read samples from configured channels.

        Args:
            num_samples: Number of samples to read per channel
            reshape_output: If True, reshapes output for multiple channels
                          into (num_channels, num_samples) format

        Returns:
            numpy.ndarray: Array of read samples
        """
        self.task.timing.cfg_samp_clk_timing(
            clk_rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=num_samples,
        )
        data = self.task.read(READ_ALL_AVAILABLE)

        if reshape_output and len(self.channels) > 1:
            return np.array(data).reshape((len(self.channels), num_samples))
        return np.array(data)

    def get_channel_info(self, channel: str) -> dict:
        """
        Get information about a specific channel.

        Args:
            channel: Channel identifier

        Returns:
            dict: Channel information including limits and name
        """
        channel_name = f"{self.device_name}/{channel}"
        if channel_name in self.channels:
            ch = self.channels[channel_name]
            return {
                "name": ch.physical_channel.name,
                "max_val": ch.ai_max,
                "min_val": ch.ai_min,
            }
        return None

    def start_continuous_acquisition(
        self,
        callback_function: Callable,
        sample_rate: float = 1000.0,
        samples_per_callback: int = 1000,
    ) -> None:
        """
        Start continuous data acquisition with callback function.

        Args:
            callback_function: Function to process acquired data
            sample_rate: Sample rate in Hz
            samples_per_callback: Number of samples to acquire before calling callback
        """

        def internal_callback(
            task_handle, every_n_samples_event_type, number_of_samples, callback_data
        ):
            """Internal callback that maintains total sample count and calls user callback."""
            self.total_samples_read += number_of_samples
            data = self.task.read(number_of_samples_per_channel=number_of_samples)
            callback_function(data, self.total_samples_read)
            return 0

        self.total_samples_read = 0
        self.task.timing.cfg_samp_clk_timing(
            sample_rate, sample_mode=AcquisitionType.CONTINUOUS
        )
        self.task.register_every_n_samples_acquired_into_buffer_event(
            samples_per_callback, internal_callback
        )
        self._continuous_acquisition_running = True
        self.task.start()

    def stop_continuous_acquisition(self) -> None:
        """Stop continuous data acquisition."""
        if self._continuous_acquisition_running:
            self.task.stop()
            self._continuous_acquisition_running = False

    def generate_waveform(
        self,
        frequency: float,
        amplitude: float,
        offset: float,
        sampling_rate: float,
        number_of_samples: int,
        phase_in: float = 0.0,
        waveform_type: str = "sine",
        duty_cycle: float = 0.5,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate a waveform with specified parameters.

        Args:
            frequency: Frequency of the waveform in Hz
            amplitude: Peak amplitude of the waveform
            offset: DC offset of the waveform
            sampling_rate: Sampling rate in Hz
            number_of_samples: Number of samples to generate
            phase_in: Initial phase in radians
            waveform_type: Type of waveform ('sine', 'square', 'sawtooth', 'triangle')
            duty_cycle: Duty cycle for square wave (0 to 1)

        Returns:
            Tuple containing the generated waveform array and final phase
        """
        duration_time = number_of_samples / sampling_rate
        duration_radians = duration_time * 2 * np.pi
        phase_out = (phase_in + duration_radians) % (2 * np.pi)
        t = np.linspace(
            phase_in, phase_in + duration_radians, number_of_samples, endpoint=False
        )

        if waveform_type == "sine":
            waveform = np.sin(frequency * t)
        elif waveform_type == "square":
            waveform = signal.square(frequency * t, duty=duty_cycle)
        elif waveform_type == "sawtooth":
            waveform = signal.sawtooth(frequency * t)
        elif waveform_type == "triangle":
            waveform = signal.sawtooth(frequency * t, width=0.5)
        else:
            raise ValueError(f"Unsupported waveform type: {waveform_type}")

        return amplitude * waveform + offset, phase_out

    def start_continuous_generation(
        self,
        waveform_params_list: List[dict],
        sampling_rate: float = 1000.0,
        number_of_samples: int = 1000,
    ) -> None:
        """
        Start continuous waveform generation on configured analog output channels.

        Args:
            waveform_params_list: List of dictionaries containing waveform parameters for each channel
            sampling_rate: Sampling rate in Hz
            number_of_samples: Number of samples per waveform cycle
        """
        self.task.timing.cfg_samp_clk_timing(
            sampling_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
        )

        actual_sampling_rate = self.task.timing.samp_clk_rate
        print(f"Actual sampling rate: {actual_sampling_rate:g} S/s")

        # Initialize numpy array for waveforms
        num_channels = len(self.channels)
        waveforms = np.zeros((num_channels, number_of_samples))

        # Generate waveform for each channel
        for i, chn in enumerate(self.channels):
            waveform, _ = self.generate_waveform(
                sampling_rate=actual_sampling_rate,
                number_of_samples=number_of_samples,
                **waveform_params_list[i],
            )
            waveforms[i] = waveform

        # Reshape to 1D if only one channel
        if num_channels == 1:
            waveforms = waveforms.reshape(-1)

        self.task.write(waveforms)
        self.task.start()
        self._continuous_generation_running = True

    def stop_continuous_generation(self) -> None:
        """Stop continuous waveform generation."""
        if self._continuous_generation_running:
            self.task.stop()
            self._continuous_generation_running = False
