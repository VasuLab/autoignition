import cantera as ct
import numpy as np
import numpy.typing as npt


class Simulation:
    def __init__(self, gas: ct.Solution | str, T: float, P: float, X: str | dict[str, float]):
        """
        Args:
            gas: Cantera gas phase object or filepath to mechanism.
            T: Temperature [K].
            P: Pressure [Pa].
            X: Species mole fractions.

        """

        self.gas = gas if isinstance(gas, ct.Solution) else ct.Solution(gas)
        self.gas.TPX = T, P, X

        self.reactor = ct.Reactor(self.gas)
        self.reactor_net = ct.ReactorNet([self.reactor])
        self.states = ct.SolutionArray(self.gas, extra=["t"])

        self.states.append(self.reactor.thermo.state, t=0.0)  # Add initial state

    def run(self, t: float = 1):
        """
        This function returns `self` to support call chaining.

        Args:
            t: Simulation end time [s].
        """
        while self.reactor_net.time < t:
            self.reactor_net.step()
            self.states.append(self.reactor.thermo.state, t=self.reactor_net.time)

        return self

    def save(self, filepath: str) -> str:
        """
        !!! Note
            If `filepath` doesn't end with ".yaml", it will be appended automatically.

        !!! Warning
            `save` will overwrite an existing file with the same filepath.

        Args:
            filepath: Filepath to save the state data.

        Returns:
            Filepath of saved state.
        """
        if not filepath.endswith(".yaml"):
            filepath += ".yaml"
        self.states.save(filepath, name="simulation", overwrite=True)
        return filepath

    @classmethod
    def restore(cls, filepath: str, mech: str):
        """
        !!! Warning
            Integrator state is not preserved; therefore, continuing a simulation from saved
            state data, although it will yield essentially the same results, will not *exactly* match a 
            simulation ran without saving and reloading, as the integrator must take smaller steps initally. 

        Args:
            filepath: Filepath to the saved state data in YAML format.
            mech: Mechanism file to reinitialize the simulation and state.
        """
        gas = ct.Solution(mech)
        states = ct.SolutionArray(gas)
        states.restore(filepath, name="simulation")

        T, P, X = states[-1].TPX
        sim = cls(gas, T, P, X)
        sim.states = states
        sim.reactor_net.initial_time = states.t[-1]
        sim.reactor_net.reinitialize()

        return sim

    @property
    def t(self) -> npt.NDArray[np.float64]:
        """Reactor elapsed time [s]."""
        return self.states.t

    @property
    def T(self) -> npt.NDArray[np.float64]:
        """Reactor temperature history [K]."""
        return self.states.T

    @property
    def P(self) -> npt.NDArray[np.float64]:
        """Reactor pressure history [Pa]."""
        return self.states.P

    def X(self, species: str) -> npt.NDArray[np.float64]:
        """        
        Args:
            species: Name of species.

        Returns:
            Mole fraction history for the `species`.
        """
        return self.states(species).X.flatten()

    def ignition_delay_time(self, species: str | None = None, *, method: str = "inflection") -> float | None:
        """
        Calculates the ignition delay time from the reactor temperature history, or `species` mole fraction if given,
        using the specified `method`.

        !!! Note
            Returns `None` if the ignition delay time is at the end of the simulation.

        Args:
            species: Name of species.
            method:
                Method used to calculate ignition delay time.

                  - 'inflection' (max slope)
                  - 'max'

        Returns:
            Ignition delay time [s].
        """

        x = self.T if species is None else self.X(species)
        if method == "inflection":
            i = np.argmax(np.diff(x) / np.diff(self.t))
            return self.t[i] if i != len(self.t) - 2 else None
        elif method == "max":
            i = np.argmax(x)
            return self.t[i] if i != len(self.t) - 1 else None
        else:
            raise ValueError(
                f"Invalid method '{method}'; valid methods are 'inflection' and 'peak'."
            )

    def get_top_species(self, n: int | None = None, *, exclude: str | list[str] | None = None) -> list[str]:
        """
        Args:
            n: Number of species to include - all species are included by default.
            exclude: Species to exclude.

        Returns:
            List of top `n` species by mole fraction in descending order.
        """

        X_max = np.max(self.states.X.T, axis=1)
        species = [
            t[1] for t in sorted(zip(X_max, self.states.species_names), reverse=True)
        ]

        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]
            for s in exclude:
                try:
                    species.remove(s.upper())
                except ValueError:
                    pass

        return species[:n]

