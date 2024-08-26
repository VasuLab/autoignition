import numpy as np
from matplotlib import pyplot as plt


class Figure:
    """
    Class for creating ignition delay time figures with the standard layout:

    - Inverse temperature (1000/T) x-axis (bottom)
    - Log-scale IDT y-axis
    - Secondary temperature x-axis (top)

    Attributes:
        ax_temp: Temperature axis.
        ax_inv: Inverse temperature axis.
    """

    def __init__(self):
        # Create inverse temperature axis
        _, self.ax_inv = plt.subplots()
        self.ax_inv.set_yscale("log")

        # Set default bounds
        self.ax_inv.set_xlim(1000.0 / 1500, 1.0)
        self.ax_inv.set_ylim(5e-5, 5e-3)

        # Create temperature axis
        def convert(x: float) -> float:
            return 1000.0 / x

        self.ax_temp = self.ax_inv.secondary_xaxis("top", functions=(convert, convert))

        # Add labels
        self.ax_inv.set_ylabel(r"Ignition Delay Time [s]")
        self.ax_inv.set_xlabel(r"1000 K/T")
        self.ax_temp.set_xlabel(r"Temperature [K]")

    def plot_exp(self, T: list[float] | np.ndarray, IDT: list[float] | np.ndarray, **kwargs):
        """
        Args:
            T: Temperatures [K].
            IDT: Ignition delay times [s].
        """
        T = np.asarray(T)  # Convert to array for division
        return self.ax_inv.scatter(1000.0 / T, IDT, **kwargs)

