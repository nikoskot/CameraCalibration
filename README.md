# Camera Calibration

## Overview

This project provides Python scripts for calibrating monocular and stereo camera setups using OpenCV. It supports both live capture from cameras and pre-captured images of a chessboard calibration pattern. The calibration process estimates camera intrinsic parameters (camera matrix and distortion coefficients) and, for stereo setups, the extrinsic parameters (rotation and translation between cameras).

## Prerequisites

- Python 3.x
- Required Python packages:
  - opencv-python
  - numpy
  - tqdm
  - plotly
  - configargparse
  - pyyaml
  - rerun-sdk

Install the dependencies using pip:

```bash
pip install opencv-python numpy tqdm plotly configargparse pyyaml rerun-sdk
```

## Calibration Pattern

The calibration uses a chessboard pattern. The default configuration expects:
- 9 rows of corners (inner corners)
- 6 columns of corners (inner corners)

You can modify these parameters in the configuration files.

## Monocular Camera Calibration

Monocular calibration estimates the intrinsic parameters of a single camera.

### Using Pre-captured Images

1. Place your calibration images in the `calibrationImages` folder. Prefix with "left_" or "right_" and set `imagesGroup` in the config.

2. Edit `monocularCalibrationConfig.yaml` to adjust settings if needed (see Configuration section below).

3. Run the calibration script:

   ```bash
   python monocularCameraCalibration.py
   ```

### Live Capture from Camera

1. Set `liveCapture: true` in `monocularCalibrationConfig.yaml`.

2. Run the calibration script:

   ```bash
   python monocularCameraCalibration.py
   ```

3. In the camera window:
   - Press 's' to start capturing images automatically every 3 seconds.
   - Press 'q' to stop capturing and proceed with calibration.

The script will detect chessboard corners in the images, perform calibration, and display a reprojection error scatter plot.

### Results

Calibration results are saved in a timestamped folder under `monocularCalibrationResults/` (e.g., `20260420_113704/`):
- `calib.json`: Calibration parameters (camera matrix, distortion coefficients, RMSE)
- `config.yaml`: Configuration used for this run
- `annotatedImages/`: Images with detected corners annotated

## Stereo Camera Calibration

Stereo calibration estimates intrinsic parameters for both cameras and extrinsic parameters (rotation and translation between them).

### Using Pre-captured Images

1. Place your calibration images in the `calibrationImages` folder.
   - Images must be paired: `left_001.png`, `right_001.png`, `left_002.png`, `right_002.png`, etc.

2. Edit `stereoCalibrationConfig.yaml` to adjust settings if needed.

3. Run the calibration script:

   ```bash
   python stereoCameraCalibration.py
   ```

### Live Capture from Two Cameras

1. Set `liveCapture: true` in `stereoCalibrationConfig.yaml`.

2. Run the calibration script:

   ```bash
   python stereoCameraCalibration.py
   ```

3. In the camera windows:
   - Press 's' to start capturing synchronized images from both cameras every 3 seconds.
   - Press 'f' to swap left/right camera assignments if needed.
   - Press 'q' to stop capturing and proceed with calibration.

The script performs monocular calibration for each camera first, then stereo calibration. It also launches a 3D visualization using rerun showing the camera positions and orientations.

### Results

Calibration results are saved in a timestamped folder under `stereoCalibrationResults/`:
- `stereoCalib.json`: Stereo calibration parameters (camera matrices, distortion coefficients, rotation, translation, essential/fundamental matrices, RMSE)
- `config.yaml`: Configuration used for this run
- `annotatedImages/`: Images with detected corners annotated

## Configuration

Both scripts use YAML configuration files for settings:

### monocularCalibrationConfig.yaml
- `imagesFolder`: Path to folder containing calibration images (default: ".\calibrationImages")
- `liveCapture`: Whether to capture images live from camera (default: false)
- `imagesGroup`: For grouped images, "left" or "right" (default: "left")
- `patternRowCorners`: Number of inner corner rows in chessboard (default: 9)
- `patternColumnCorners`: Number of inner corner columns in chessboard (default: 6)
- `dontRefineCorners`: Whether to skip subpixel corner refinement (default: false)
- `resultsSavePath`: Base path for saving results (default: ".\monocularCalibrationResults")

### stereoCalibrationConfig.yaml
- `imagesFolder`: Path to folder containing calibration images (default: ".\calibrationImages")
- `liveCapture`: Whether to capture images live from cameras (default: False)
- `patternRowCorners`: Number of inner corner rows in chessboard (default: 9)
- `patternColumnCorners`: Number of inner corner columns in chessboard (default: 6)
- `patternGridSize`: Size of chessboard squares in mm (default: 1)
- `dontRefineCorners`: Whether to skip subpixel corner refinement (default: false)
- `resultsSavePath`: Base path for saving results (default: ".\stereoCalibrationResults")

You can override these settings via command-line arguments. Use `--help` to see available options.

## Troubleshooting

- Ensure your chessboard pattern is clearly visible in all images.
- For stereo calibration, both cameras should see the pattern simultaneously.
- If calibration fails, check that the pattern detection is working (annotated images will show detected corners).
- For live capture, ensure cameras are accessible (check camera indices in code if needed).

