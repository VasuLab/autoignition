import cantera as ct
from matplotlib import pyplot as plt
import numpy as np


class AutoignitionSimulation(ct.ReactorNet):
    def __init__(self, reactor: ct.Reactor):
        self.reactor = reactor
        super().__init__([self.reactor])
        self.states = ct.SolutionArray(reactor.thermo, extra=["t"])

    def step(self):
        super().step()
        self.states.append(self.reactor.thermo.state, t=self.time)

    def ignition_delay_time(
        self, species: str = None, *, method: str = "inflection"
    ) -> float:
        """
        Calculates the ignition delay time from the reactor temperature history, or species mole fraction if given,
        using the specified method.

        !!! Warning
            Returns [`np.nan`](https://numpy.org/doc/stable/reference/constants.html#numpy.nan) if calculated
            ignition delay time occurs at the end of the simulated time.

        Args:
            species: Species name.
            method: Method used to calculate ignition delay time.

                  - "inflection" (default) or max slope

                  - "max"

        Returns:
            Ignition delay time [s].

        """

        x = self.states.T if species is None else self.states(species).X.flatten()
        if method == "inflection":
            i = np.argmax(np.gradient(x, self.states.t))
            return self.states.t[i] if i != len(self.states.t) - 2 else np.nan
        elif method == "max":
            i = np.argmax(x)
            return self.states.t[i] if i != len(self.states.t) - 1 else np.nan
        else:
            raise ValueError(
                f"Invalid method '{method}'; valid methods are 'inflection' and 'peak'."
            )

    def get_top_species(
        self, n: int = None, *, exclude: str | list[str] = None
    ) -> list[str]:
        """
        Returns the top `n` species by mole fraction in descending order. If `n` is not given,
        all non-excluded species are returned.

        Args:
            n: Number of species (optional).
            exclude: Species to exclude (optional).

        Returns:
            List of top species.

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


gas = ct.Solution("gri30.yaml")
gas.TPX = 1000, 10e5, "H2: 0.1, O2: 0.05, Ar: 0.85"
sim = AutoignitionSimulation(ct.Reactor(gas))
sim.advance_to_steady_state()

print(sim.get_top_species(10, exclude=["AR"]))

plt.plot(sim.states.t, sim.states.T)
plt.axvline(sim.ignition_delay_time())
plt.show()
