# Heston-Bates Option Pricing and Calibration

This project implements the Heston-Bates model for vanilla option pricing. It extends the classical Heston model by including jump components. The code uses Fourier inversion techniques and least squares optimization to calibrate the model parameters to market data.

## Features

- **Characteristic Function:** Computes the characteristic function combining the Heston stochastic volatility core with a jump adjustment.
- **Option Pricing via FFT:** Prices European call and put options using Fourier transform inversion.
- **Calibration:** Calibrates the model parameters (volatility, mean reversion, jump intensity, etc.) to market data extracted from a DataFrame.
- **Market Data Processing:** Extracts strike and volatility information from market quotes and constructs the required data structure.

## Project Structure

- **heston-bates.ipynb:** The main Jupyter Notebook that contains the model implementation, calibration routines, and market data extraction.
- **93.ipynb:** Contains the Heston 1993 model implementation, calibration routines, and data extraction.
- **README.md:** This file provides an overview and documentation for the project.

## Getting Started

1. Install the required Python packages:
   - NumPy
   - Pandas
   - SciPy
   - Matplotlib (for visualizations)
   - SciPy Stats (for Black–Scholes pricing)

2. Run the notebook and follow the cell instructions:
   - Import libraries.
   - Define model functions (characteristic function, option pricing, calibration).
   - Extract and prepare market data.
   - Calibrate the Heston-Bates model and compare its prices against Black–Scholes benchmarks.

## Model Overview

The Heston-Bates model extends the Heston model by accounting for jumps in the underlying asset price. The characteristic function is modified with a jump component, and the drift is adjusted accordingly. The model is calibrated using market prices through non-linear least squares minimization.

For more details on the mathematical formulation, refer to the documentation cells within the notebook.
