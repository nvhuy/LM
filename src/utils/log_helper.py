'''
Created on Dec 24, 2017

@author: HuyNguyen
'''
import logging
from logging.handlers import RotatingFileHandler
import datetime
import os
from src.utils import fs_helper

LOG_FORMAT = '%(levelname)-8s %(asctime)s %(thread)d %(module)s:%(lineno)s %(funcName)s %(message)s'
LOG_FORMAT_SHORT = '%(levelname)-8s %(funcName)s %(message)s'


def now_to_string(dateonly=False):
    datetime_fmt = '%Y%m%d-%H%M%S'
    if dateonly:
        datetime_fmt = '%Y%m%d'
    return datetime.datetime.now().strftime(datetime_fmt)


def generate_log_name(log_file_name, dateonly=False, unique_id=False):
    """
    Create file name for log data
    :param log_file_name: log file
    :param dateonly: if true, log file is named using date
    :param unique_id: if true, log file has named attached with uuid
    """
    log_file_name = log_file_name + '_' + now_to_string(dateonly) + '.log'
    if unique_id:
        log_file_name = fs_helper.add_uuid_to_file_name(log_file_name)
    return log_file_name


def create_rotating_log_handler(log_file):
    """
    Set-up a rotating file log handler
    :param log_file: log file
    """
    rotate_handler = RotatingFileHandler(filename=log_file, mode='a', maxBytes=11 * 1024 * 1024,
                                         backupCount=1000000, encoding='UTF-8', delay=True)
    log_formatter = logging.Formatter(LOG_FORMAT)
    rotate_handler.setFormatter(log_formatter)
    rotate_handler.setLevel(logging.INFO)
    return rotate_handler


def create_console_log_handler(console_logging_level=logging.ERROR):
    """
    Log handler to print to stdout
    """
    # create console handler with a higher log level
    log_formatter = logging.Formatter(LOG_FORMAT_SHORT)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_logging_level)
    console_handler.setFormatter(log_formatter)
    return console_handler


def setup_root_logger(log_file_name, date_only=False, unique_id=False, console_logging_level=logging.ERROR,
                      log_file_directory=None):
    """
    Set up logging configuration
    :param log_file_name: name of log file
    :param date_only: if set then log file is named using only date but not time
    :param unique_id: if true, log file has named attached with uuid
    :param console_logging_level: console logging level
    :param log_file_directory: log folder
    """
    root_logger = logging.getLogger('')
    root_logger.setLevel(logging.INFO)
    log_file_name_ext = generate_log_name(log_file_name, date_only, unique_id)

    rotate_handler = None
    if log_file_directory is not None:
        fs_helper.fs_make_folder(log_file_directory)
        log_file = os.path.join(log_file_directory, log_file_name_ext)
        rotate_handler = create_rotating_log_handler(log_file)
        root_logger.addHandler(rotate_handler)
    else:
        return None, None
    if rotate_handler is not None:
        root_logger.addHandler(rotate_handler)

    console_handler = create_console_log_handler(console_logging_level)
    root_logger.addHandler(console_handler)

    return root_logger, rotate_handler
