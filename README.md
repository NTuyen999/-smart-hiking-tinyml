# Smart Hiking Band - TinyML Fall Detection

TinyML research and prototype for fall detection
on the ESP32-based Smart Hiking Band.

## Dataset

BITS-2 Fall Detection Dataset
https://doi.org/10.5281/zenodo.10013090

## Current experiments

- CNN v1: initial baseline
- CNN v2: additional ADL training windows
- Input: 40 x 3 accelerometer samples
- Model architecture: 1D CNN, 1001 parameters
- Split: subject-independent Train / Validation / Test

## Important limitations

- FALL windows currently use candidate labels.
- Assumed sampling frequency: 20 Hz.
- ESP32 firmware currently samples at 50 Hz.
- Test set has not yet been evaluated.
- The models are not ready for safety-critical alerts.

## Status

Experimental models only.
ESP32 integration and real-world validation are pending.

## Reproducibility note
V1 preprocessing and training scripts are now included. The reconstructed preprocessing matches all 1,629 original windows within a tolerance of 1e-4. Exact retraining results are not guaranteed. FALL labels remain peak-centered candidates; final independent testing and ESP32 validation are pending.

