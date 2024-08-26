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
        prop_groups: Property groups.
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

        # Property groups
        self.prop_groups: dict[str, dict] = {}

    # Note: Merging dictionaries with `a | b` replaces values from `a` with `b` in the case of duplicate keys.
    # This behavior is utilized in the plotting functions to override properties on a priority basis.

    def plot_exp(self, T: list[float] | np.ndarray, IDT: list[float] | np.ndarray, *groups, **kwargs):
        """
        Args:
            T: Temperatures [K].
            IDT: Ignition delay times [s].
            *groups: Group names.
        """
        props = {}

        # Apply relevant property groups
        if groups is not None:
            print("Hello")
            for g in groups:
                try:
                    props = props | self.prop_groups[g]
                except KeyError:
                    raise ValueError(f"Invalid property group name: '{g}'")

        props = props | kwargs  # Override properties with keyword arguments

        T = np.asarray(T)  # Convert to array for division
        return self.ax_inv.scatter(1000.0 / T, IDT, **props)

    def show(self):
        plt.show()
