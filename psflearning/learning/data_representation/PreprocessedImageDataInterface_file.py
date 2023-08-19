from abc import ABCMeta, abstractmethod
import pickle


class PreprocessedImageDataInterface:
    """
    Interface that ensures consistency and compatability between all old and new implementations of data classes, fitters and psfs.
    Classes implementing this interafce should hold preprocessed image data and provide needed information to fitters and psfs.
    In general, we use the following convention for the dimensions of image data. This can be extended by adding dimensions on the left.
        channels, images/rois, z, y, x
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_image_data(self) -> list:
        """
        Is called from a fitter or a psf and returns a list containing the needed image information.
        In general, these are the cropped rois, the centers of the rois and in some
        cases a list of indices that indicate from which image the roi was cut.
        """
        raise NotImplementedError("You need to implement a 'get_image_data' method in your data representation class.")

    def save(self, filename: str) -> None:
        """
        Save object to file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(filename: str):
        """
        Load object from file.
        """
        with open(filename, "rb") as f:
            self = pickle.load(f)
        return self