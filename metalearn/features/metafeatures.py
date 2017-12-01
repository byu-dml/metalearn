import numpy as np

class Metafeature(object):

    def __init__(self):
        pass

    """
    compute is the general purpose function to implement for a new metafeature or group of metafeatures

    Parameters
    ----------
    X: The data for metafeatures to be computed on
    Y: The labels for the data, ordered so it lines up with X
    attributes: The names of the attributes in X and Y
    Returns
    -------
    A list of 1 or more metafeatures. Because metafeatures sometimes are computed together, this 
    will always be a list, although it may just be a list of 1, since we'd like to split these up 
    as small as possible
    """
    def compute(self, X: list, Y: list, attributes: list) -> list:        
        raise NotImplementedError("Cannot call compute on MetaFeature object")
