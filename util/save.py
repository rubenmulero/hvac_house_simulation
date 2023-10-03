"""
This file will be used to store data in a database, or CSV files.

Data will be received in a Python like dict and store it.

"""

import os
import csv

from util.logger import LoggingClass

# logger = LoggingClass('save_data').get_logger()


def save_to_csv(p_data, p_path):
    """
    By giving data and a destination path, this method stores the obtained data in a CSV file format.

    :param p_data: a Python list of dicts
    :param p_path: the full path (with file name) to save the file

    :return:
    """
    assert type(p_data) is dict
    filename, file_extension = os.path.splitext(p_path)
    assert file_extension == '.csv'
    keys = p_data[0].keys()
    with open(p_path, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(p_data)
    print("Data stored ok")


def write_progress(p_data, p_path):
    """
    This method is used to write the current simulation progress in disk. The idea is to have a tracking of each
    decision taken from the developed agent

    :param p_data: The data to be stored in CSV file in Python dictionary format.
    :param p_path: the full path (with file name) to save the file

    :return:
    """
    assert type(p_data) is dict
    filename, file_extension = os.path.splitext(p_path)
    assert file_extension == '.csv'
    # Creating the directory path in case that do not exist.
    file_exists = os.path.isfile(p_path)
    with open(p_path, 'a', newline='') as csvfile:
        headers = list(p_data.keys())
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        # Writing fields
        writer.writerow(p_data)
