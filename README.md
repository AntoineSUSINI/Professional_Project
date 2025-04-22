# Heston-Bates Option Pricing and Calibration

This project implements various option pricing models based on the Heston framework. It extends the classical Heston model by including jump components, explores different calibration methodologies, and even tackles multi-factor volatility models.

## Features

- **Characteristic Functions:** Compute characteristic functions for the Heston-Bates and Heston 1993 models.
- **Option Pricing via FFT:** Price European call and put options using Fourier inversion techniques.
- **Calibration Procedures:** Calibrate model parameters (stochastic volatility, jump intensity, etc.) through least squares optimization.
- **Market Data Processing:** Extract and structure market data from raw quotes for model calibration.
- **Model Comparisons:** Analyze and compare model outputs (e.g., Black–Scholes, Heston-Bates, Heston 1993, and Double Heston).

## Project Structure

- **heston-bates.ipynb:** Main notebook with model implementation, calibration routines, and market data extraction based on the Heston-Bates model.
- **93.ipynb:** Notebook containing the Heston 1993 model implementation, calibration routines, and data extraction.
- **double_heston.ipynb:** Notebook for the Double Heston model, including calibration and performance comparison.
- **README.md:** This file that provides an overview and documentation of the project.

## Getting Started

1. Install the required Python packages:
   - NumPy
   - Pandas
   - SciPy
   - Matplotlib (for visualizations)
   - SciPy Stats (for Black–Scholes pricing)

2. Run the notebooks and follow the cell instructions:
   - Import libraries.
   - Define model functions (characteristic function, option pricing, calibration).
   - Extract and prepare market data.
   - Calibrate the models and compare their outputs against benchmarks.

## Model Overview

The project explores several models based on the Heston framework. The Heston-Bates model incorporates jumps, while the Heston 1993 model and the Double Heston model offer alternative calibrations and volatility dynamics. Detailed explanations and mathematical formulations can be found within each notebook’s documentation cells.
