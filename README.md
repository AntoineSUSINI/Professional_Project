# Option Pricing Models and Web Interface

This project implements various option pricing models based on the Heston framework and provides a web interface for easy model calibration and option pricing. It includes the classical Heston model, Heston-Bates with jumps, and a Double Heston variant.

## Features

### Models Implemented
- **Heston 1993:** Classical stochastic volatility model 
- **Heston-Bates:** Extends Heston with jump components
- **Double Heston:** Two-factor stochastic volatility model
- **Web Interface:** Flask-based UI for model calibration and pricing

### Core Functionality
- **Characteristic Functions:** Implementations for all model variants
- **FFT-based Pricing:** Fast Fourier Transform for European options
- **Model Calibration:** Least squares optimization using market data
- **Market Data Processing:** Excel data import and preprocessing
- **Interactive UI:** Simple web interface for model usage

## Project Structure

### Model Implementation
- **Heston93.ipynb:** Base Heston model implementation
- **heston-bates.ipynb:** Heston-Bates model with jumps
- **double_heston.ipynb:** Two-factor volatility model

### Web Application
- **app.py:** Flask server implementation
- **templates/:** HTML templates
  - **index.html:** Main web interface
- **static/:** Web assets
  - **style.css:** UI styling
  - **script.js:** Frontend logic

## Getting Started

### Dependencies
```bash
pip install numpy pandas scipy matplotlib flask
```

### Running the Web Interface
1. Start the Flask server:
```bash
python app.py
```
2. Open http://localhost:5000 in your browser
3. Upload market data Excel file
4. Use calibrated model for option pricing

### Using the Notebooks
1. Open the desired notebook (Heston93.ipynb, heston-bates.ipynb, or double_heston.ipynb)
2. Follow cell-by-cell execution for:
   - Model implementation
   - Data preparation
   - Calibration
   - Option pricing

## Market Data Format

The Excel file should contain:
- Implied volatility surface
- Risk-free rate
- Dividend yield
- Spot price
- Various strikes and maturities

See example market data files for the expected format.

## Models Overview

### Heston 1993
- Single stochastic volatility factor
- Mean-reverting variance process
- Correlation between asset and volatility

### Heston-Bates
- Extends Heston with jump process
- Additional parameters for jump intensity and size
- Better fits market skew

### Double Heston
- Two volatility factors
- Each factor follows Heston dynamics
- Improved fit for both short and long maturities
