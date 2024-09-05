from concurrent.futures import ProcessPoolExecutor, Future

import cantera as ct
import numpy as np
import numpy.typing as npt
import os


class Simulation:
    def __init__(
        self, gas: ct.Solution | str, T: float, P: float, X: str | dict[str, float]
    ):
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

    def ignition_delay_time(
        self, species: str | None = None, *, method: str = "inflection"
    ) -> float | None:
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

    def get_top_species(
        self, n: int | None = None, *, exclude: str | list[str] | None = None
    ) -> list[str]:
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


class SimulationPool:
    def __init__(self, max_workers: int | None = None, output_dir: str | None = None):
        """
        Args:
            max_workers: Maximum number of worker processes.
            output_dir: Directory to output simulation files.
        """
        self._max_workers = max_workers
        self.executor: ProcessPoolExecutor | None = None
        self.futures: dict[int, Future] = {}
        self.parameters: dict[int, dict] = {}
        self._simulation_count: int = 0

        self._output_dir = None
        self.output_dir = output_dir if output_dir is not None else "output"

    def __enter__(self):
        # Create the ProcessPoolExecutor when entering the context
        self.executor = ProcessPoolExecutor(max_workers=self._max_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Ensure that the executor is properly shut down when exiting
        if self.executor:
            self.executor.shutdown(wait=True)
        self.executor = None

    def __getitem__(self, id: int) -> Simulation:
        try:
            filepath = self.futures[id].result()
        except KeyError:
            raise ValueError("Invalid simulation ID.")

        mech = self.parameters[id]["mech"]
        return Simulation.restore(filepath, mech)

    @property
    def output_dir(self) -> str | None:
        return self._output_dir

    @output_dir.setter
    def output_dir(self, output_dir: str):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self._output_dir = output_dir

    @staticmethod
    def _run_simulation(mech: str, T: float, P: float, X, output_filepath: str) -> str:
        sim = Simulation(mech, T, P, X)
        sim.run()
        filepath = sim.save(output_filepath)
        return filepath

    def submit_simulation(
        self, mech: str, T: float, P: float, X, *, filename: str | None = None
    ) -> int:
        """
        Args:
            mech: Filepath to Cantera mechanism.
            T: Temperature [K].
            P: Pressure [Pa].
            X: Species mole fractions.
            filename: Name of the output simulation file (default: `sim[id].yaml`).

        Returns:
            id: Simulation ID number.
        """

        if self.executor:
            id = self._simulation_count
            self._simulation_count += 1

            self.parameters[id] = {"mech": mech, "T": T, "P": P, "X": X}
            self.futures[id] = self.executor.submit(
                self._run_simulation,
                mech,
                T,
                P,
                X,
                os.path.join(
                    self.output_dir,
                    filename if filename is not None else f"sim{id}.yaml",
                ),
            )

            return id

        raise RuntimeError("ProcessPoolExecutor not initialized")
