# Title: preprocessing.py
# Date: 31/01/227
# Author: James Barrett

# Description: Adopts the preprocessing steps defined by Liao et al. (2019) for KDSB17

def main(slices):
    """
    :param slices: list of numpy.ndarray for each slice
    :return: processed scan images as uint16
    """
    # Do model implemention with no preprocessing except done by DAZA et al. (2021)