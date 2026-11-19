import numpy as np

def flip_x(image):
    """
    Flip an image across the x-axis (vertical flip).

    Parameters
    ----------
    image : np.ndarray
        Image array of shape (H, W) or (H, W, C)

    Returns
    -------
    np.ndarray
        Flipped image
    """
    return np.flipud(image)