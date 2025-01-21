# Shrink Calibration Plugin for Pioreactor

This plugin provides functionality to shrink multiple calibrations across workers in a Pioreactor cluster. It uses a regularized optimization process to fit a unified polynomial model while adjusting for worker-specific variations.

## Features

- Collects calibration data from multiple workers in a Pioreactor cluster.
- Fits a regularized polynomial model.
- Allows user interaction to select calibrations and customize parameters.
- Automatically distributes the resulting calibration back to the workers.

## Prerequisites

Before using this plugin, ensure you have:

- A Pioreactor cluster with workers set up and active.
- Calibrations already recorded on each worker for the specified device.
- The Pioreactor software stack installed on your system.

## Installation

There are three options:
1. Install this plugin on your leader via the UI's Plugin page (requires internet access)
2. Install this plugin on your leader via `pio plugins install pioreactor-calibration-shrinkage` (requires internet access)
3. Or copy the `__init__.py` file's contents into a new `.py` file in the `~/.pioreactor/plugins` directory on your leader.


## Usage

### Running the Plugin

Use the following command to start the shrink calibration process:

```bash
pioreactor calibrations shrinkage --device <DEVICE>
```

Replace `<DEVICE>` with the device you want to shrink calibrations for (e.g., `pump`, `sensor`).

### Workflow

1. **Calibration Selection**: The plugin will list available calibrations on each worker. Select the calibrations you wish to include in the shrinkage process.
2. **Parameter Configuration**:
   - **Polynomial Degree**: Specify the degree of the polynomial for the unified model.
   - **Closeness Parameter**: Adjust the regularization parameter (`lambda_a`) to control how closely individual calibrations align with the average.
3. **Model Fitting**: The plugin fits the model and displays the results. If you're not satisfied, you can adjust parameters and refit.
4. **Calibration Distribution**: Once satisfied, the plugin distributes the new calibration data back to the workers.

### Example

```bash
pioreactor calibrations shrinkage --device media_pump
```

- Follow the prompts to select calibrations and configure the fitting parameters.
- The plugin will output logs detailing the progress and results.

## How It Works

1. **Data Collection**: The plugin retrieves calibration data from each worker using the Pioreactor API.
2. **Model Fitting**: It fits a polynomial model using regularized optimization, adjusting worker-specific scalars (`A`) and the polynomial coefficients (`w`).
3. **Calibration Creation**: A new calibration is created for each worker, incorporating the unified model and worker-specific adjustments.
4. **Calibration Distribution**: The new calibration is posted back to each worker for use.


## Support

If you encounter any issues or have questions, please visit the Pioreactor community forums or contact support.

---

Happy calibrating! 🎉
