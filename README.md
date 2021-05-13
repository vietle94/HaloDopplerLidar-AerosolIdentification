# HaloDopplerLidar-ClassificationAlgorithm

From Viet Le thesis in Helsinki University titled "Aerosol depolarization ratio at 1565nm in Finland with Halo Doppler lidar"

# Required modules
- xarray
- sklearn
- scipy
- numpy
- matplotlib
- argparse
- json

The algorithm is run in a terminal window
# Run the algorithm on one file
python classifier.py "Path-to-input-file" "Path-to-output-directory"

Example:

```python
python classifier.py "F:/halo/46/depolarization/20171017_fmi_halo-doppler-lidar-46-depolarization.nc" "C:/Users/vietl/Desktop"
```

# Run the algorithm on multiple file
python classifier_multiple.py "Path-to-input-directory" "Path-to-output-directory"

Example:

```python
python classifier_multiple.py "F:/halo/46/depolarization/" "C:/Users/vietl/Desktop"
```


# Run the algorithm on XR-Uto file
python classifier.py "Path-to-input-file" "Path-to-output-directory" -XR

Example:

```python
python classifier.py "F:/halo/32/depolarization/20171017_fmi_halo-doppler-lidar-46-depolarization.nc" "C:/Users/vietl/Desktop" -XR
```
