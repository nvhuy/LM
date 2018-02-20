"""
Created on Dec 29, 2017

@author: HuyNguyen
"""
import os
import subprocess
import logging
import uuid
import codecs
import sys
import re
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


def get_lm_file_from_data_file(data_file):
    """
    Calculate language model file from data file name
    """
    data_file_name = os.path.basename(data_file)
    return data_file_name + '.lm'


def parse_ngram_output(ngram_output):
    """
    Read output of ngram to extract log probability and perplexity
    :param ngram_output: output of `ngram_file`
    """
    module_logger.info('------ Parsing ngram output file')

    res = []

    output_lines = ngram_output.split('\n')
    n = len(output_lines)
    i = 0
    while i + 3 < n:
        sent = output_lines[i].strip()
        output = output_lines[i + 2].strip().split()
        try:
            logprob = float(output[3])
        except:
            logprob = -999999.0
        try:
            ppl = float(output[5])
        except:
            ppl = sys.float_info.min
        try:
            ppl1 = float(output[7])
        except:
            ppl1 = sys.float_info.min
        res.append((sent, logprob, ppl, ppl1))
        i += 4

    return res


class SriLm:

    def __init__(self, ngram_count_file=None, ngram_file=None):
        """
        Set up SRI folder and executable files
        :param ngram_count_file: path to ngram-count
        :param ngram_file: path to ngram file
        """
        self.__ngram_count_file__ = ngram_count_file
        self.__ngram_file__ = ngram_file

    def set_ngram_path(self, ngram_file=None):
        """
        Set up SRI folder and executable files
        :param ngram_file: path to ngram file
        """
        self.__ngram_file__ = ngram_file

    def set_ngram_count_path(self, ngram_count_file=None):
        """
        Set up SRI folder and executable files
        :param ngram_count_file: path to ngram-count
        """
        self.__ngram_count_file__ = ngram_count_file

    def ngram_count_file(self, data_file, lm_file, ngram_order, sos=True):
        """
        Run ngram_count for a given text file
        :param data_file: input text file
        :param lm_file: path to language model file
        :param ngram_order: ngram order
        :param sos: if True, insert sos and eos symbols
        """
        module_logger.info('------ Run ngram_count ::: {}'.format(data_file))
        cmd = [self.__ngram_count_file__,
               '-unk', '-tolower', '-interpolate',
               '-order', str(ngram_order),
               '-text', data_file,
               '-lm', lm_file]
        if not sos:
            cmd.extend(['-no-sos', '-no-eos'])
        try:
            _, err, _ = run_system_cmd(cmd)
        except:
            module_logger.exception('****** Failed running ngram-count')
            err = -1
        return err

    def train_language_models(self, sample_folder, name_pattern, lm_folder, ngram_order, sos=True):
        """
        Create a language model for responses of each prompt
        :param sample_folder: folder to save responses of prompts
        :param name_pattern: specify file names of interest
        :param lm_folder: folder to save language models
        :param ngram_order: ngram order
        :param sos: if True, insert sos and eos symbols
        """
        if not os.path.exists(lm_folder):
            os.makedirs(lm_folder)

        for data_file_name in os.listdir(sample_folder):
            data_file = os.path.join(sample_folder, data_file_name)
            if os.path.isfile(data_file) and re.match(name_pattern, data_file_name) is not None:
                lm_file_name = os.path.basename(data_file_name) + '.lm'
                lm_file_path = os.path.join(lm_folder, lm_file_name)
                self.ngram_count_file(data_file, lm_file_path, ngram_order, sos)

    def ngram_file(self, data_file, lm_file, ngram_order, sos=True):
        """
        Run ngram to estimate probability of text in a file
        :param data_file: file of input text
        :param lm_file: language model file
        :param ngram_order: ngram order
        :param sos: if True, insert sos and eos symbols
        """
        module_logger.info('------ Run ngram ::: {}'.format(data_file))

        cmd = [self.__ngram_file__,
               '-tolower', '-debug', '1', '-order', str(ngram_order), '-ppl', data_file, '-lm', lm_file]
        if not sos:
            cmd.extend(['-no-sos', '-no-eos'])

        res = None
        try:
            output, _, _ = run_system_cmd(cmd)
            res = parse_ngram_output(output)
        except:
            module_logger.exception('****** Failed running ngram')

        return res

    def ngram_text(self, input_text, lm_file, ngram_order, sos=True):
        """
        Run ngram to estimate probability of text
        :param input_text: input text as a list of lines
        :param lm_file: language model file
        :param ngram_order: ngram order
        :param sos: if True, insert sos and eos symbols
        """
        lm_folder = os.path.dirname(lm_file)
        temp_file_path = os.path.join(lm_folder, '__' + uuid.uuid4().urn[9:])
        try:
            temp_file = codecs.open(temp_file_path, mode='wb', encoding='utf-8')
            temp_file.write('\n'.join(input_text))
            temp_file.close()
            output = self.ngram_file(temp_file_path, lm_file, ngram_order, sos)
        except:
            output = None
        finally:
            os.remove(temp_file_path)

        return output
