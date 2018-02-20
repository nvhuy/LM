"""
Created on Dec 29, 2017

@author: HuyNguyen
"""
import os
import subprocess
import logging
module_logger = logging.getLogger(__name__)


def run_system_cmd(args_list):
    """
    Run system command using `subprocess`
    :param args_list: list of command arguments, including command name at the beginning
    :return: output, error message, exit code
    """
    print('\n[>>>>] running system command: {0}'.format(' '.join(args_list)))
    prc = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = prc.communicate()
    print 'Exit code', prc.returncode
    if prc.returncode:
        print('\n:::::: return code: %d, error: %s' % (prc.returncode, errors))
        # raise RuntimeError('Error running command: %s. Return code: %d, Error: %s' % (' '.join(args_list),
        #                                                                               prc.returncode, errors))
    return output, errors, prc.returncode


class ParseText:

    def __init__(self, parse_lib_folder, parse_config_file):
        """
        Set up parse library folder and configuration file
        :param parse_lib_folder: lib folder
        :param parse_config_file: config file
        """
        self.__parse_lib_folder__ = parse_lib_folder
        self.__parse_config_file__ = parse_config_file

    def parse(self, parse_folder):
        """
        Parse file in folder
        :param parse_folder: input folder
        """
        module_logger.info('------ Parse folder ::: %s' % parse_folder)

        cmd = ['java', '-Xmx7g', '-Dfile.encoding=UTF-8', '-classpath', os.path.join(self.__parse_lib_folder__, '*'),
               'parseText.parseText', self.__parse_config_file__, parse_folder]
        _, emsg, err = run_system_cmd(cmd)
        if err != 0:
            module_logger.error('****** Failed parsing ::: %s' % emsg)
        return err
