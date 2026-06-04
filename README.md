# SEIHRV Epidemic Modeling

A mathematical and computational investigation of infectious disease dynamics using an extended SEIHRV framework incorporating environmental transmission, hospitalization, and vaccination.

## Overview

This project studies the spread of infectious diseases through a nonlinear compartmental model consisting of six interacting populations:

* **S** — Susceptible
* **E** — Exposed
* **I** — Infectious
* **H** — Hospitalized
* **R** — Recovered
* **V** — Environmental Viral Load

Unlike classical SIR and SEIR models, the SEIHRV framework captures indirect transmission through environmental reservoirs, allowing for more realistic modeling of diseases where pathogens persist outside the host population.

The project combines mathematical analysis, numerical simulation, and visualization to investigate equilibrium behavior, disease persistence, environmental feedback effects, and long-term epidemic outcomes.

## Research Objectives

* Develop a nonlinear epidemic model incorporating environmental transmission.
* Analyze disease-free and endemic equilibria.
* Derive and interpret the basic reproduction number (R_0).
* Investigate stability conditions and bifurcation behavior.
* Examine environmental reservoirs as drivers of sustained transmission.
* Visualize epidemic trajectories through phase portraits and simulations.

## Mathematical Model

The model is governed by a system of coupled differential equations:

```math
\frac{dS}{dt} = ...
```

representing transitions between susceptible, exposed, infectious, hospitalized, recovered, and environmental compartments.

Transmission occurs through both:

1. Direct contact with infectious individuals.
2. Indirect exposure through environmental contamination.

This environmental feedback mechanism introduces richer dynamics than standard compartmental epidemic models.

## Key Topics Investigated

### Basic Reproduction Number

The next-generation matrix approach is used to derive:

```math
R_0 = \rho(FV^{-1})
```

where:

* (F) is the transmission matrix.
* (V) is the transition matrix.
* (\rho) denotes the spectral radius.

### Equilibrium Analysis

The project examines:

* Disease-Free Equilibrium (DFE)
* Endemic Equilibrium

and determines conditions under which each equilibrium exists and remains stable.

### Stability Analysis

Methods employed include:

* Jacobian matrices
* Eigenvalue analysis
* Lyapunov techniques
* Center manifold methods

to characterize local and global behavior.

### Numerical Simulation

Numerical solutions illustrate:

* Epidemic outbreaks
* Long-term persistence
* Oscillatory behavior
* Environmental amplification effects
* Sensitivity to parameter changes

## Repository Contents

```text
SEIHRV/
│
├── Paper.pdf
├── Presentation.pdf
├── Figures/
├── final
├── Simulations/
├── Source_Code/
├── README.md
```

## Results

The model demonstrates that environmental reservoirs can significantly alter epidemic dynamics by:

* Extending outbreak duration
* Increasing peak infection levels
* Supporting endemic persistence
* Producing complex nonlinear trajectories

Simulation results reveal parameter regimes in which environmental transmission becomes a dominant contributor to disease spread.

## Visualizations

The repository includes:

* Time-series epidemic trajectories
* Phase portraits
* Equilibrium diagrams
* Sensitivity analyses
* Three-dimensional state-space visualizations

## Skills Demonstrated

* Mathematical modeling
* Differential equations
* Dynamical systems
* Epidemiology
* Scientific computing
* Numerical simulation
* Research communication
* Data visualization
* LaTeX

## Author

**Pramithas Upreti**

Mathematics • Computer Science • Logic

## Citation

If referencing this project, please cite the accompanying manuscript included in this repository.

## Acknowledgements

This work was completed as part of an undergraduate mathematical modeling research project exploring nonlinear epidemic dynamics and environmental transmission pathways. Mathematical techniques were informed by literature on compartmental epidemiological models, next-generation matrix methods, and stability theory. Related compartmental modeling frameworks, such as SEIR and generalized epidemic models, provided a theoretical background.
