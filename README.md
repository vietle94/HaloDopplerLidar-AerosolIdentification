# HaloDopplerLidar-ClassificationAlgorithm


# Required modules
- xarray
- sklearn
- scipy
- numpy
- matplotlib
- argparse
- json

# Run the algorithm on one file


```python
python classifier.py "F:/halo/46/depolarization/20171017_fmi_halo-doppler-lidar-46-depolarization.nc" "C:/Users/vietl/Desktop"
```


# Run the algorithm on multiple file
python classifier_multiple.py "F:/halo/46/depolarization/" "C:/Users/vietl/Desktop"

# Run the algorithm on XR-Uto file
python classifier.py "F:/halo/32/depolarization/20171017_fmi_halo-doppler-lidar-46-depolarization.nc" "C:/Users/vietl/Desktop" -XR
