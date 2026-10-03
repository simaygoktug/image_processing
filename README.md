# Image Processing

Classical image processing experiments with OpenCV: vibration detection from camera images, window-based floor counting, and a visual-words feature pipeline for machine fault diagnosis.

## Overview

The scripts apply traditional computer vision (filtering, frequency analysis, wavelets, keypoint detection, contour analysis and clustering) to engineering problems. Two reference papers on image-based vibration analysis and machine fault diagnosis are included as background.

## Contents

| Path | Description |
| --- | --- |
| `vibration_detection_and_analysis_with_image_processing/main.py` | High-pass filtering, histogram equalisation, FFT magnitude spectrum, continuous wavelet transform and SIFT keypoints; compares a test image against a reference image with SSIM, PSNR and MSE and flags excessive vibration against thresholds |
| `predicting_floor_number_according_to_window_detection.py` | Google Colab script: Canny edges, morphological closing and contour filtering detect window-like rectangles, then DBSCAN on their y positions estimates the number of floors |
| `classification.py` | Draft visual-words pipeline: integral image, SURF descriptors, k-means vocabulary and histogram feature vector (no trained classifier yet, and it relies on a `load_image` helper defined in `main.py`) |
| `*.pdf` | Reference papers: "An Image Processing Approach to Machine Fault Diagnosis Based on Visual Words Representation" and "Computer Vision Tracking Techniques Applied to Vibration Analysis" |

## Tech stack

- Python, OpenCV, NumPy, SciPy, scikit-image, scikit-learn, Matplotlib
- Google Colab (floor counting script)

## How to run

Vibration analysis expects `test_image.png` and `reference_image.png` in the working directory:

```bash
pip install opencv-python numpy scipy scikit-image matplotlib
cd vibration_detection_and_analysis_with_image_processing
python main.py
```

`scipy.signal.cwt` and `ricker` were removed in SciPy 1.15, so use an earlier SciPy version. The floor counting script uses `google.colab` for file upload and display and is meant to be run in a Colab notebook. SURF in `classification.py` requires an `opencv-contrib-python` build with non-free modules enabled.

## Author

Goktug Can Simay: [GitHub](https://github.com/simaygoktug) | [Website](https://goktugcansimay.com)
