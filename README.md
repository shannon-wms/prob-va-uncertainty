# Probabilistic Volcanic Ash Uncertainty

Incorporating source parameter (MER and plume height) alongside meteorological variability in volcanic ash hazard dispersion forecasting.

Contains code submitted alongside manuscript "Incorporating Source Parameter and Meteorological Variability in the Generation of Probabilistic Volcanic Ash Hazard Forecasts" to Journal of Geophysical Research: Atmospheres.

Requires The Met Office's [Numerical Atmospheric-dispersion Modelling Environment](https://www.metoffice.gov.uk/research/approach/modelling-systems/dispersion-model) (NAME), which is available by licence from the UK Met Office.

## Introduction

Airborne volcanic ash is hazardous for aircraft. To manage this risk, Volcanic Ash Advisory Centres (VAACs) provide forecasts of ash clouds following a volcanic eruption. These forecasts are created using dispersion models that predict the transport of ash based on eruption details and weather data. These sets of inputs have large uncertainties that can affect the accuracy of the forecasts. The paper this project is associated with presents a method for producing probabilistic forecasts that account for these uncertainties. Typically, weather uncertainties are handled by using multiple weather predictions, referred to as an ensemble. Dispersion outcomes depend on the eruption plume height and mass eruption rate (MER), which are related but have large associated uncertainties. Our method uses a statistical approach to incorporate these uncertainties into forecasts to allow for the calculation of probabilities of different ash concentration levels for aviation. It uses the same number of model runs as there are ensemble members, and does not require eruption details (plume height, MER, and emission profile) to be specified in advance, making it a computationally efficient approach as the bulk of computations can be done after a small number of initial model runs. 

## Installation

It is recommended to clone this repository and create a virtual environment before installing the package dependencies.

```bash
# Clone the repository
git clone git@github.com:shannon-wms/prob-va-uncertainty.git

# Navigate to directory
cd prob-va-uncertainty

# Set up Conda environment
conda create --name pva python=3.8

# Activate virtual environment
conda activate pva

# Install dependencies
pip install -r requirements.txt

# Install pvauncertainty package
pip install -e .
```
## Usage

The Jupyter notebooks `get-started-pt1` and `get-started-pt2` illustrate how the package can be used with NAME. Users must provide their own NAME input files to simulate volcanic ash dispersion; minimal non-working examples of code block segments that must be changed are given in `scripts`.

## Contents

### pvauncertainty

The package contains classes to set up volcanic ash simulations in NAME and evaluate resultant probabilistic quantities:
  + Set up NAME inputs for a volcanic ash emission given a plume height observation, or range for the height:
    + Using deterministic or ensemble met
    + Provides a unit MER for later rescaling
    + Given a plume height range and interval step size, initialises NAME with multiple interval emissions to be saved separately
    + Sets NAME running on a SLURM environment
  + Evaluate probabilistic quantities of volcanic ash concentrations:
    + Conditional exceedance probabilities given ensemble member
    + Conditional exceedance probabilities given plume height observation
    + Percentiles of ash concentration given plume height observation
    + Overall exceedance probabilities given plume height distribution (Gaussian distribution by default) by numerical integration (quadrature)
  + Plotting of probabilistic quantities of volcanic ash concentrations

### scripts

Contains scripts for setting up ensemble or deterministic NAME runs, given a csv file of plume height and MER values, and minimal example NAME input files.

### analysis

Contains scripts for post-processing of data and generation of figures for the submitted manuscript.

## License

This project is licensed under the MIT License.

## Contact 

Any queries contact shannon.williams@bristol.ac.uk.