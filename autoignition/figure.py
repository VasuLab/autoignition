import numpy as np
from matplotlib import pyplot as plt
import uncertainties
from uncertainties import unumpy


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

    exp_props = {"linestyle": "None", "marker": "o", "capsize": 5}
    """Default properties for errorbar plots."""

    sim_props = {}
    """Default properties for all simulation line plots."""

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

    def _get_group_props(self, groups: list[str]) -> dict:
        props = {}
        for g in groups:
            try:
                props = props | self.prop_groups[g]
            except KeyError:
                raise ValueError(f"Invalid property group name: '{g}'")
        return props

    def plot_exp(
        self,
        T: list[float] | np.ndarray,
        IDT: list[float] | np.ndarray,
        *groups,
        **kwargs,
    ):
        """
        Args:
            T: Temperatures [K].
            IDT: Ignition delay times [s].
            *groups: Property group names.
        """
        props = self.exp_props
        props = props | self._get_group_props(groups)  # Override properties with group properties
        props = props | kwargs  # Override properties with keyword arguments

        T_inv = 1000.0 / np.asarray(T)  # Invert temperature
        IDT = np.asarray(IDT)

        T_uncertainty = T_inv.dtype == uncertainties.core.Variable
        IDT_uncertainty = IDT.dtype == uncertainties.core.Variable

        return self.ax_inv.errorbar(
            unumpy.nominal_values(T_inv) if T_uncertainty else T_inv, 
            unumpy.nominal_values(IDT) if IDT_uncertainty else IDT, 
            unumpy.std_devs(IDT) if IDT_uncertainty else None,
            unumpy.std_devs(T_inv) if T_uncertainty else None,
            **props
        )
    
    def plot_sim(
        self, 
        T: list[float] | np.ndarray, 
        IDT: list[float] | np.ndarray, 
        *groups,
        **kwargs
    ):
        """
        Args:
            T: Temperatures [K].
            IDT: Ignition delay times [s].
            *groups: Property group names.
        """

        props = self.sim_props
        props = props | self._get_group_props(groups)  # Override properties with group properties
        props = props | kwargs  # Override properties with keyword arguments

        T_inv = 1000.0 / np.asarray(T)
        return self.ax_inv.plot(T_inv, IDT, **props)

    def show(self):
        plt.show()
