"""
This is a NUMPY parser to  check and load data in a numpy data format.


@author: Rubén Mulero

"""


from pathlib import Path
import numpy as np

from util.logger import LoggingClass

logger = LoggingClass('context_data').get_logger()


class NUMPYParser:

    def __init__(self, p_path):
        self.path = Path(p_path)

    def parse_data(self, p_file):
        """
        Using the provided file, this method loads its contents in a numpy float64 object

        :param p_file: The name of the file to load

        :return: A numpy object with dtype of float64 with the loaded data or None if nothing detected
        """
        data = None
        if p_file and self._is_csv(p_file):
            # Loading file using pandas
            logger.info("Valid CSV file found, loading it")
            csv_file = self.path / p_file
            data = np.loadtxt(str(csv_file), dtype='float64', delimiter=',')
            data = data.reshape(1, len(data))
        return data

    def _is_csv(self, p_file):
        """
        Checks if the current loaded file is a CSV file.

        :param p_file: A file name
        :return: True if it is a CSV file, False otherwise.
        """
        res = False
        if self._check_file(p_file):
            # Seems to be a valid file, checking extension
            if p_file.endswith('.csv'):
                res = True
        return res

    def _check_file(self, p_file):
        """
        This method checks if the provided file exist or not in the system

        :param p_file: A file name
        :return: True if the file exists in the current path, False otherwise
        """
        res = False
        if self._check_path():
            my_file = self.path / p_file
            if my_file.is_file():
                # File exists
                res = True
            else:
                print("The provided file do not exist in the system: {path}".format(path=my_file))
                logger.warning("The provided file do not exist in the system: {path}".format(path=my_file))
        return res

    def _check_path(self):
        """
        This method checks if the provided path in the constructor exist or not in the system

        :return: True if it is a path, else False
        """
        res = False
        if self.path.is_dir():
            res = True
        else:
            print("The provided path do not exist in the system: {path}".format(path=self.path))
            logger.warning("The provided path do not exist in the system: {path}".format(path=self.path))
        return res
