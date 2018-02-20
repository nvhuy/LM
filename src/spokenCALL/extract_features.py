"""
Created on Dec 28, 2017

@author: HuyNguyen
"""
import argparse
import os
import logging
import codecs
import json
import re
import time

module_logger = logging.getLogger(__name__)

from src.utils import log_helper, nltk_helper, gensim_helper, \
    generic_helper, srilm_helper, spell_helper, parse_helper
from src.spokenCALL import read_grammar, ml, feature_names

project_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_folder = os.path.join(project_folder, 'logs')
_, GLOBAL_rotate_handler = log_helper.setup_root_logger('lm', date_only=True, unique_id=False,
                                                        console_logging_level=logging.INFO,
                                                        log_file_directory=log_folder)


class StudentResponses:

    def __init__(self):
        self.__student_responses__ = {}
        self.__feature_names__ = []

        self.__reference_grammar__ = None

        self.__head_pattern_list__ = generic_helper.create_head_word_patterns()
        self.__filler_pattern_list__ = generic_helper.create_filler_sound_patterns()
        self.__edit_patterns__ = None

    def read_grammar_from_json(self, grammar_file):
        """
        Read sample responses in reference grammar
        """
        module_logger.info('------ Load JSON dump of reference grammar ::: {}'.format(grammar_file))
        try:
            del self.__reference_grammar__
            jfile = open(grammar_file.replace('.xml', '.json'))
            self.__reference_grammar__ = json.load(jfile)
            jfile.close()
        except:
            module_logger.exception('****** Failed loading JSON dump of reference grammar')
            self.__reference_grammar__ = None

    def read_data(self, training_file, no_duplicate=False, no_trace=True, use_transcript=False, clean_res=False):
        """
        Read training data into a dictionary
        {prompt : [{id, language, meaning, response}]}
        :param training_file: training file
        :param no_duplicate: if True, remove duplicates in data
        :param no_trace: if True, ignore annotation trace
        :param use_transcript: if True, overwrite ASR with human transcript
        :param clean_res: if True, do cleaning text responses
        """
        module_logger.info('------ Read data ::: {} ::: {} ::: {}'.format(training_file, use_transcript, clean_res))

        try:
            del self.__feature_names__
            del self.__student_responses__
        except:
            module_logger.exception('****** Class variables not initialized')
        finally:
            self.__feature_names__ = []
            self.__student_responses__ = dict()

        try:
            tf = codecs.open(training_file, mode='rb', encoding='utf-8', errors='ignore')
            '''ignore the header line'''
            headers = tf.readline().strip().replace('"', '').split('\t')

            id_idx = -1
            if 'Id' in headers:
                id_idx = headers.index('Id')
            pmp_idx = -1
            if 'Prompt' in headers:
                pmp_idx = headers.index('Prompt')

            if id_idx == -1 or pmp_idx == -1:
                module_logger.error('****** Invalid data file. No ID or prompt field')
                return

            asr_idx = -1
            if 'RecResult' in headers:
                asr_idx = headers.index('RecResult')

            trs_idx = -1
            if 'Transcription' in headers:
                trs_idx = headers.index('Transcription')

            if asr_idx == -1 and trs_idx == -1:
                module_logger.error('****** Invalid data file. No ASR result & transcription')
                return

            lng_idx = -1
            if 'language' in headers:
                lng_idx = headers.index('language')
            mng_idx = -1
            if 'meaning' in headers:
                mng_idx = headers.index('meaning')
            response_language = 'correct'
            response_meaning = 'correct'

            trace_unreliable_list = ['M: 0-4 H: 1-0', 'M: 0-4 H: 2-1', 'M: 4-0 H: 0-1', 'M: 4-0 H: 1-2']
            trc_idx = -1
            if 'Trace' in headers:
                trc_idx = headers.index('Trace')
            trace_code = 'M: 0-0 H: 0-0'

            print id_idx, pmp_idx, asr_idx, trs_idx, lng_idx, mng_idx, trc_idx
            read_response = {}

            response_line = tf.readline().strip().replace('"', '')
            while response_line != '':
                response_tup = response_line.split('\t')
                '''read the next line'''
                response_line = tf.readline().strip().replace('"', '')

                response_id = response_tup[id_idx]
                dutch_prompt = response_tup[pmp_idx]

                if asr_idx > -1:
                    response_asr = response_tup[asr_idx]
                else:
                    response_asr = response_tup[trs_idx]

                if trs_idx > -1:
                    response_trans = response_tup[trs_idx]
                else:
                    response_trans = response_asr

                if lng_idx > -1:
                    response_language = response_tup[lng_idx]
                if mng_idx > -1:
                    response_meaning = response_tup[mng_idx]

                '''skip unreliable code'''
                if trc_idx > -1:
                    trace_code = response_tup[trc_idx]
                if not no_trace and trace_code in trace_unreliable_list:
                    continue

                asr_tokens = response_asr.split()
                phn_count = len(asr_tokens)
                unk_count = 0.0
                for tok in asr_tokens:
                    if tok.startswith('*'):
                        unk_count += 1

                if clean_res:
                    response_asr = generic_helper.clean_transcript(response_asr,
                                                                   self.__filler_pattern_list__,
                                                                   self.__head_pattern_list__)
                else:
                    response_asr = generic_helper.remove_unrecognized_tokens(response_asr)
                if response_asr == '':
                    response_asr = '? ? ? ? ? ? ? ? ? ? ? ? ?'

                if clean_res:
                    response_trans = generic_helper.clean_transcript(response_trans,
                                                                     self.__filler_pattern_list__,
                                                                     self.__head_pattern_list__)
                else:
                    response_trans = generic_helper.remove_unrecognized_tokens(response_trans)
                if response_trans == '':
                    response_trans = '? ? ? ? ? ? ? ? ? ? ? ? ?'

                if dutch_prompt not in self.__student_responses__:
                    self.__student_responses__[dutch_prompt] = []
                    read_response[dutch_prompt] = set()

                if no_duplicate and response_asr in read_response[dutch_prompt]:
                    continue
                else:
                    read_response[dutch_prompt].add(response_asr)

                self.__student_responses__[dutch_prompt].append({'id': response_id,
                                                                 'sounds': phn_count,
                                                                 'unknown': unk_count,
                                                                 'response': response_asr,
                                                                 'transcript': response_trans,
                                                                 'language': response_language,
                                                                 'meaning': response_meaning,
                                                                 'features': []})
            if use_transcript:
                self.use_transcripts()
        except:
            module_logger.exception('****** Failed reading training data')
            self.__student_responses__ = {}

    def use_transcripts(self):
        """
        Overwrite responses with transcripts
        Should only call right inside read data function
        """
        module_logger.info('------ Use transcript to replace provided response text')
        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                res['response'] = res['transcript']

    def add_translated_prompt(self):
        """
        Add translated prompts
        """
        module_logger.info('------ Add translated prompts')
        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                res['prompt_list'] = [' '.join(prom['tokens'])
                                      for prom in self.__reference_grammar__[dutch_prompt]['prompt_list']]

    def get_prompt_distribution(self, data_file):
        """
        :param data_file: output file
        """
        module_logger.info('------ Get prompt distribution')

        df = codecs.open(data_file.replace('.csv', '_prm.dis'), mode='wb', encoding='utf-8')
        for dutch_prompt in self.__student_responses__:
            translated_prompt = ' '.join(self.__reference_grammar__[dutch_prompt]['prompt_list'][0]['tokens'])
            incorrect_count = 0.0
            correct_count = 0.0
            for res in self.__student_responses__[dutch_prompt]:
                if res['language'] == 'correct':
                    correct_count += 1.0
                else:
                    incorrect_count += 1.0
            df.write('\t'.join([translated_prompt, str(correct_count), str(incorrect_count),
                                str(correct_count / (correct_count + incorrect_count))])
                     + '\n')
        df.close()

    def add_external_asr(self, asr_file, clean_asr=False):
        """
        Add student responses from external ASR output.
        :param asr_file: ASR output file with response id and response text
        :param clean_asr: if True, clean response text
        """
        module_logger.info('------ Add external ASR output to replace provided response text ::: {}'.format(asr_file))

        external_asr = {}
        try:
            asrf = codecs.open(asr_file, mode='rb', encoding='utf-8')
            for asr_line in asrf.readlines():
                asr_tup = asr_line.strip().lower().split(',')
                if len(asr_tup) != 2:
                    continue
                if clean_asr:
                    cleaned_asr = generic_helper.clean_transcript(asr_tup[1], self.__filler_pattern_list__,
                                                                  self.__head_pattern_list__)
                else:
                    cleaned_asr = generic_helper.remove_unrecognized_tokens(asr_tup[1])
                external_asr[asr_tup[0]] = cleaned_asr
        except:
            module_logger.exception('****** Failed reading ASR output file ::: {}'.format(asr_file))
            return

        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                res_id = res['id']
                if res_id in external_asr:
                    res['response'] = external_asr[res_id]
                    if res['response'] == '':
                        res['response'] = '? ? ? ? ? ? ? ? ? ? ? ? ?'

    def add_external_asr_nbest(self, asr_file, clean_asr=False):
        """
        Add student responses from external ASR output.
        :param asr_file: ASR output file with response id and response text
        :param clean_asr: if True, clean response text
        """
        module_logger.info('------ Add external ASR output to replace provided response text ::: {}'.format(asr_file))

        external_asr_nbest = {}
        try:
            asrf = codecs.open(asr_file, mode='rb', encoding='utf-8')
            for asr_line in asrf.readlines():
                asr_tup = asr_line.strip().lower().split(',')
                if len(asr_tup) != 2:
                    continue
                asr_id = asr_tup[0].split('-')[0]
                if asr_id not in external_asr_nbest:
                    external_asr_nbest[asr_id] = []
                if clean_asr:
                    cleaned_asr = generic_helper.clean_transcript(asr_tup[1], self.__filler_pattern_list__,
                                                                  self.__head_pattern_list__)
                else:
                    cleaned_asr = generic_helper.remove_unrecognized_tokens(asr_tup[1])
                external_asr_nbest[asr_id].append(cleaned_asr)
        except:
            module_logger.exception('****** Failed reading ASR output file ::: {}'.format(asr_file))
            return

        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                res_id = res['id']
                if res_id in external_asr_nbest:
                    res['response-nbest'] = external_asr_nbest[res_id]

        for dutch_prompt in self.__student_responses__:
            if self.__reference_grammar__ is not None:
                sample_list = self.__reference_grammar__[dutch_prompt]['response_list']
            else:
                sample_list = None
            for res in self.__student_responses__[dutch_prompt]:
                if 'response-nbest' not in res:
                    continue
                most_common = 0.0
                best_res = res['response-nbest'][0]
                if sample_list is not None:
                    for nbest_txt in res['response-nbest']:
                        nbest_common1 = 0.0
                        for spl in sample_list:
                            sample_response = spl['tokens']
                            common1 = nltk_helper.common_ngrams(nbest_txt.split(), sample_response, 1, False)
                            if common1 > nbest_common1:
                                nbest_common1 = common1
                        if nbest_common1 > most_common:
                            most_common = nbest_common1
                            best_res = nbest_txt
                res['response'] = best_res
                if res['response'] == '':
                    res['response'] = '? ? ? ? ? ? ? ? ? ? ? ? ?'

    def calculate_wer(self):
        """
        Calculate Word-Error-Rate
        """
        module_logger.info('------ Calculate WER')

        num_del = 0.0
        num_ins = 0.0
        num_sub = 0.0
        len_total = 0.0
        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                rep = res['response']
                if rep.startswith('?'):
                    rep = ''
                    continue
                trs = res['transcript']
                if trs.startswith('?'):
                    trs = ''
                    continue
                wer = generic_helper.calculate_wer(trs, rep)
                if trs != '':
                    len_total += len(trs.split())
                num_del += wer['Del']
                num_ins += wer['Ins']
                num_sub += wer['Sub']

        print '[INFO] WER', (num_del + num_sub + num_ins) / len_total

    def write_grammar_as_training(self, new_training_file):
        """
        Combine sample responses in reference grammar and incorrect responses in training data
        :param new_training_file: new training file path
        """
        module_logger.info('------ Build training data from reference grammar ::: {}'.format(new_training_file))

        tf = codecs.open(new_training_file, mode='wb', encoding='utf-8')
        tf.write('Id\tPrompt\tRecResult\tTranscription\tlanguage\tmeaning\n')

        for dutch_prompt in self.__student_responses__:
            '''write correct responses in reference grammar'''
            if dutch_prompt in self.__reference_grammar__:
                response_list = self.__reference_grammar__[dutch_prompt]['response_list']
                for res in response_list:
                    tup = [res['id'], dutch_prompt, res['text'], res['text'], 'correct', 'correct']
                    tf.write('\t'.join(tup) + '\n')
            else:
                module_logger.warning('****** Prompt in training not found in reference grammar')

            '''write incorrect responses in training data'''
            for res in self.__student_responses__[dutch_prompt]:
                if res['language'] == 'incorrect':
                    tup = [res['id'], dutch_prompt, res['response'], res['transcript'], res['language'], res['meaning']]
                    tf.write('\t'.join(tup) + '\n')
        tf.close()

    def write_responses_to_parse(self, data_file, parse_folder, transcription=False):
        """
        Write responses to file for parsing
        :param parse_folder: folder with input files for parser
        :param data_file: output file
        :param transcription: if True, write transcription
        """
        module_logger.info('------ Write responses to parse ::: {}'.format(data_file))

        data_file_base = os.path.splitext(os.path.basename(data_file))[0]
        if transcription:
            data_file_base = data_file_base + '-trs'
        parse_text_folder = os.path.join(parse_folder, data_file_base)
        if not os.path.exists(parse_text_folder):
            os.makedirs(parse_text_folder)
        module_logger.info('------ Parse text folder ::: {}'.format(parse_text_folder))

        rid_file = codecs.open(os.path.join(parse_folder, data_file_base + '.id'), mode='wb', encoding='utf-8')
        res_file = codecs.open(os.path.join(parse_text_folder, data_file_base + '.res'), mode='wb', encoding='utf-8')

        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                rid_file.write(res['id'] + '\n')
                if transcription:
                    res_file.write(res['transcript'] + '\n')
                else:
                    res_file.write(res['response'] + '\n')

        rid_file.close()
        res_file.close()

        return parse_text_folder

    def read_responses_from_parse(self, data_file, parse_folder, transcription=False):
        """
        Read parse output of responses
        :param data_file: response file
        :param parse_folder: folder of parsing output
        :param transcription: if True, read parse of transcripts
        """
        module_logger.info('------ Read parse output of responses ::: {}'.format(data_file))

        data_file_base = os.path.splitext(os.path.basename(data_file))[0]
        if transcription:
            data_file_base = data_file_base + '-trs'
        parse_file_base = os.path.join(parse_folder, data_file_base)

        id_file = parse_file_base + '.id'
        res_file = parse_file_base + '.parsed'
        response_parsed_dict = generic_helper.read_responses_from_parse(res_file, id_file)

        if response_parsed_dict is None:
            module_logger.error('****** Failed reading parse output file')
            return

        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                response_id = res['id']
                if response_id in response_parsed_dict:
                    response_parsed = response_parsed_dict[response_id]
                    if transcription:
                        prefx = 'trs-'
                    else:
                        prefx = ''
                    for parse_feat in response_parsed:
                        res[prefx + parse_feat] = response_parsed[parse_feat]
                        if parse_feat == 'parse':
                            res[prefx + 'prod'] = nltk_helper.extract_production_rules(response_parsed[parse_feat],
                                                                                       include_leaf=False)
                        if parse_feat == 'dependency':
                            res[prefx + 'dep'] = generic_helper.reduce_dependency_rules(response_parsed[parse_feat])

    def collect_correct_responses(self):
        """
        Collect correct responses into a dictionary
        :return: {prompt : set[response]}
        """
        module_logger.info('------ Collect correct responses')
        correct_responses = {}
        collected_count = 0
        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                if res['language'] == 'correct' and res['meaning'] == 'correct':
                    if dutch_prompt not in correct_responses:
                        correct_responses[dutch_prompt] = set()
                    correct_responses[dutch_prompt].add(res['response'])
                    collected_count += 1
        module_logger.info('------ Correct responses ::: {}'.format(collected_count))
        return correct_responses

    def train_response_lm(self, ngram_count_file, response_folder, lm_folder, ngram_order, no_duplicate=False):
        """
        Write correct and incorrect response for LM
        Train language models for words, pos and more
        :param ngram_count_file: path to ngram-count
        :param response_folder: data output folder path
        :param lm_folder: folder path to keep LM files
        :param ngram_order: ngram order
        :param no_duplicate: if True, train on no-duplicate set
        """
        module_logger.info('------ Write transcripts for language models ::: {}'.format(response_folder))

        cw_file = codecs.open(os.path.join(response_folder, 'trs-words'), mode='wb', encoding='utf-8')
        cpos_file = codecs.open(os.path.join(response_folder, 'trs-pos'), mode='wb', encoding='utf-8')
        cprd_file = codecs.open(os.path.join(response_folder, 'trs-prod'), mode='wb', encoding='utf-8')
        cdep_file = codecs.open(os.path.join(response_folder, 'trs-dep'), mode='wb', encoding='utf-8')

        iw_file = codecs.open(os.path.join(response_folder, 'trs-words_err'), mode='wb', encoding='utf-8')
        ipos_file = codecs.open(os.path.join(response_folder, 'trs-pos_err'), mode='wb', encoding='utf-8')
        iprd_file = codecs.open(os.path.join(response_folder, 'trs-prod_err'), mode='wb', encoding='utf-8')
        idep_file = codecs.open(os.path.join(response_folder, 'trs-dep_err'), mode='wb', encoding='utf-8')

        for dutch_prompt in self.__student_responses__:
            correct_set = set()
            incorrect_set = set()
            for res in self.__student_responses__[dutch_prompt]:
                if res['language'] == 'correct':
                    if no_duplicate and res['transcript'] in correct_set:
                        continue
                    correct_set.add(res['transcript'])
                    cw_file.write(' '.join(res['trs-tokens']) + '\n')
                    cpos_file.write(' '.join(res['trs-pos']) + '\n')
                    cprd_file.write(' '.join(res['trs-prod']) + '\n')
                    cdep_file.write(' '.join(res['trs-dep']) + '\n')
                elif True or res['meaning'] == 'incorrect':
                    if no_duplicate and res['transcript'] in incorrect_set:
                        continue
                    incorrect_set.add(res['transcript'])
                    iw_file.write(' '.join(res['trs-tokens']) + '\n')
                    ipos_file.write(' '.join(res['trs-pos']) + '\n')
                    iprd_file.write(' '.join(res['trs-prod']) + '\n')
                    idep_file.write(' '.join(res['trs-dep']) + '\n')

        cw_file.close()
        cpos_file.close()
        cprd_file.close()
        cdep_file.close()

        iw_file.close()
        ipos_file.close()
        iprd_file.close()
        idep_file.close()

        sri_lm = srilm_helper.SriLm(ngram_count_file=ngram_count_file)

        '''train word models of correct and incorrect responses'''
        name_pattern = re.compile('trs-words.*$')
        sri_lm.train_language_models(response_folder, name_pattern, lm_folder, ngram_order)
        '''train POS models of correct and incorrect responses'''
        name_pattern = re.compile('trs-pos.*$')
        sri_lm.train_language_models(response_folder, name_pattern, lm_folder, ngram_order)
        '''train production rules models of correct and incorrect responses'''
        name_pattern = re.compile('trs-prod.*$')
        sri_lm.train_language_models(response_folder, name_pattern, lm_folder, 1, sos=False)
        '''train dependency rules models of correct and incorrect responses'''
        name_pattern = re.compile('trs-dep.*$')
        sri_lm.train_language_models(response_folder, name_pattern, lm_folder, 1, sos=False)

    def tag_responses(self):
        """
        Tokenize and tag student responses
        """
        module_logger.info('------ Tag student responses')
        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                response_text = res['response']
                response_tokens = nltk_helper.tokenize_sentence(response_text)
                response_pos = nltk_helper.tag_tokens(response_tokens)
                res['tokens'] = response_tokens
                res['pos'] = response_pos

    def json_dumps(self, data_file):
        """
        Dump student responses to JSON file
        """
        module_logger.info('------ Save student responses to JSON file ::: {}'.format(data_file))
        if self.__student_responses__ is None:
            module_logger.error('****** Student response is Null')
            return

        try:
            jfile = open(data_file.replace('.csv', '.json'), mode='wb')
            json.dump([self.__feature_names__, self.__student_responses__], jfile, indent=2, sort_keys=True)
            jfile.close()
            print '[INFO] response info'
            for dutch_prompt in self.__student_responses__:
                for res in self.__student_responses__[dutch_prompt]:
                    print ' *** '.join(sorted(res.keys()))
                    break
                break
        except:
            module_logger.exception('****** Failed saving student responses to JSON file')

    def json_loads(self, data_file):
        """
        Load student responses from JSON file
        """
        module_logger.info('------ Load student responses from JSON file ::: {}'.format(data_file))

        del self.__feature_names__
        del self.__student_responses__
        try:
            with open(data_file.replace('.csv', '.json'), mode='rb') as jfile:
                dat = json.load(jfile)
                self.__feature_names__ = dat[0]
                self.__student_responses__ = dat[1]
        except:
            module_logger.exception('****** Failed loading student responses from JSON file')
            self.__feature_names__ = None
            self.__student_responses__ = None

    def json_loads_more(self, data_file):
        """
        Combine data from multiple json files
        """
        module_logger.info('------ Load MORE student responses from JSON file ::: {}'.format(data_file))

        try:
            with open(data_file.replace('.csv', '.json'), mode='rb') as jfile:
                dat = json.load(jfile)
                more_student_responses = dat[1]
        except:
            module_logger.exception('****** Failed loading student responses from JSON file')
            return
        if more_student_responses is not None:
            for dutch_prompt in self.__student_responses__:
                res_list = self.__student_responses__[dutch_prompt]
                if dutch_prompt in more_student_responses:
                    res_more = more_student_responses[dutch_prompt]
                    res_list.extend(res_more)

    def write_edit_patterns(self, edit_pattern_file):
        """
        Write edit patterns to file to use in feature extraction for test date
        :param edit_pattern_file: edit pattern file
        """
        with open(edit_pattern_file, 'wb') as jf:
            json.dump(self.__edit_patterns__, jf)

    def read_edit_patterns(self, edit_pattern_file):
        """
        Read edit patterns to file to use in feature extraction for test date
        :param edit_pattern_file: edit pattern file
        """
        with open(edit_pattern_file) as jf:
            self.__edit_patterns__ = json.load(jf)

    def extract_edit_features(self, reset_edit_list=False, remove_all=False):
        """
        Edit distance features
        :param reset_edit_list: create list of edit patterns found
        :param remove_all: remove features
        """
        module_logger.info('------ Extract edit features')

        feature_columns = ['edit_distance']
        self.remove_features_prefix('edit')
        if remove_all:
            return
        self.__feature_names__.extend(feature_columns)

        se = generic_helper.SentenceDistance()

        edit_features = {}
        edit_res_id = {}

        for dutch_prompt in self.__student_responses__:
            sample_list = self.__reference_grammar__[dutch_prompt]['response_list']
            sents = []
            for res in sample_list:
                sents.append(res['pos'])

            for res in self.__student_responses__[dutch_prompt]:
                response_text = res['pos']
                min_edit, sentf = se.min_distance(response_text, sents)
                min_edit += 0.0
                res['features'].append(min_edit)

                if response_text[0] == '.':
                    edit_features[res['id']] = []
                    continue

                eds = []
                ops = se.edit_ops(response_text, sentf)
                for op in ops:
                    if op[0] == 'insert':
                        ed = 'edit_pattern-OO-' + sentf[op[3]]
                    elif op[0] == 'delete':
                        ed = 'edit_pattern-' + response_text[op[1]] + '-OO'
                    elif op[0] == 'replace':
                        ed = 'edit_pattern-' + response_text[op[1]] + '-' + sentf[op[3]]
                    else:
                        continue
                    eds.append(ed)
                    if ed not in edit_res_id:
                        edit_res_id[ed] = set()
                    edit_res_id[ed].add(res['id'])
                edit_features[res['id']] = eds

        if reset_edit_list:
            try:
                del self.__edit_patterns__
            finally:
                self.__edit_patterns__ = []
            for ed in sorted(edit_res_id.keys()):
                if len(edit_res_id[ed]) >= 5:
                    self.__edit_patterns__.append(ed)

        module_logger.info('------ Edit patterns found ::: {}'.format(len(self.__edit_patterns__)))

        self.__feature_names__.extend(self.__edit_patterns__)
        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                edits = [0.0] * len(self.__edit_patterns__)
                for ed in edit_features[res['id']]:
                    if ed in self.__edit_patterns__:
                        col = self.__edit_patterns__.index(ed)
                        edits[col] = 1
                res['features'].extend(edits)

    def extract_length_features(self, remove_all=False):
        """
        Length ratio of student response and average length of sample responses
        :param remove_all: remove features
        """
        module_logger.info('------ Extract length features')

        self.remove_features_prefix('has_response')

        feature_columns = ['length_ratio', 'length_01']
        feature_columns.extend(['length_under-min', 'length_above-max', 'length_unknown', 'length_unknown-ratio',
                                'length_sounds', 'length_sounds-ratio'])
        self.remove_features_prefix('length')
        if remove_all:
            return
        self.__feature_names__.extend(feature_columns)

        sample_length = 0.0
        for dutch_prompt in self.__student_responses__:
            sample_list = self.__reference_grammar__[dutch_prompt]['response_list']
            sample_count = len(sample_list)
            length_min = 9999.0
            length_max = 0.0
            for res in sample_list:
                lng = len(res['tokens'])
                sample_length += lng
                if lng > length_max:
                    length_max = lng
                if lng < length_min:
                    length_min = lng

            sample_length = sample_length / sample_count

            for res in self.__student_responses__[dutch_prompt]:
                response_text = res['tokens']
                unk_count = res['unknown']
                sounds = res['sounds']
                if response_text[0] == '?':
                    lng = 0
                else:
                    lng = len(response_text)
                less_min = 0.0
                if lng < length_min:
                    less_min = 1.0
                more_max = 0.0
                if lng > length_max:
                    more_max = 1.0

                length_feature = [lng / sample_length, lng / len(response_text)]
                if len(feature_columns) > 2:
                    if sounds > 0:
                        length_feature_more = [less_min, more_max, unk_count, unk_count / sounds, sounds, lng / sounds]
                    else:
                        length_feature_more = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    length_feature.extend(length_feature_more)
                res['features'].extend(length_feature)

    def extract_parse_score_features(self, remove_all=False):
        """
        Parsing score of the response
        :param remove_all: remove features
        """
        module_logger.info('------ Extract parse score')

        feature_columns = ['parse_score-ratio']
        self.remove_features_prefix('parse')
        if remove_all:
            return
        self.__feature_names__.extend(feature_columns)

        for dutch_prompt in self.__student_responses__:
            sample_list = self.__reference_grammar__[dutch_prompt]['response_list']
            sample_score_mean = 0.0
            for res in sample_list:
                sample_score_mean += res['score']
            sample_score_mean = sample_score_mean / len(sample_list)
            for res in self.__student_responses__[dutch_prompt]:
                score_feature = (res['score'] / sample_score_mean,)
                res['features'].extend(score_feature)

    def extract_spelling_features(self, remove_all=False):
        """
        Count mis-spelled words
        :param remove_all: remove features
        """
        module_logger.info('------ Extract spelling features')

        feature_columns = ['error_yn', 'error_count', 'error_ratio']
        # feature_columns.extend(['error_first', 'error_last'])
        self.remove_features_prefix('error')
        if remove_all:
            return
        self.__feature_names__.extend(feature_columns)

        ec = spell_helper.EnchantSpell()

        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                response_text = res['tokens']
                error_has = 0.0
                error_count = 0.0
                error_first = 0.0
                error_last = 0.0
                if response_text[0] == '?':
                    error_count = len(response_text) * 1.0
                    error_count_ratio = 1.0
                else:
                    error_idx = ec.check_tokens(response_text)
                    if error_idx is not None and len(error_idx) > 0:
                        error_count = 0.0 + len(error_idx)
                        if error_idx[0] == 0:
                            error_first = 1.0
                        if error_idx[-1] == len(response_text):
                            error_last = 1.0
                    error_count_ratio = error_count / len(response_text)
                if error_count > 0:
                    error_has = 1.0
                spelling_feature = [error_has, error_count, error_count_ratio]
                if len(feature_columns) > 3:
                    spelling_feature.extend([error_first, error_last])
                res['features'].extend(spelling_feature)

    def extract_prompt_features(self, remove_all=False):
        """
        Count missing prompt words
        """
        module_logger.info('------ Extract prompt features')

        self.remove_features_prefix('missing_prompt-words')

        # pos_tag_prefix = ['C', 'DT', 'EX', 'IN', 'JJ', 'LS', 'MD', 'NN', 'PR', 'RB', 'RP', 'TO', 'UH', 'VB', 'W']
        pos_tag_prefix = ['DT', 'IN', 'MD', 'NN', 'VB']

        feature_columns = ['prompt_missing', 'prompt_missing-pct']
        feature_columns.extend(['prompt_' + ptp for ptp in pos_tag_prefix])
        self.remove_features_prefix('prompt')
        if remove_all:
            return
        self.__feature_names__.extend(feature_columns)

        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                response_text = res['tokens']
                missing_count_min = 9999.0
                missing_pos_min = None
                missing_pct = 1.0
                for prom in self.__reference_grammar__[dutch_prompt]['prompt_list']:
                    prompt_pos = prom['pos']
                    prompt_tokens = prom['tokens']
                    missing_count = 0.0
                    missing_pos = [0.0] * len(pos_tag_prefix)
                    for tid, tok in enumerate(prompt_tokens):
                        if prompt_pos[tid] != '.' and tok not in response_text:
                            missing_count += 1
                            for i in range(len(pos_tag_prefix)):
                                if prompt_pos[tid].startswith(pos_tag_prefix[i]):
                                    missing_pos[i] = 1.0
                                    break
                    if missing_count_min > missing_count:
                        missing_count_min = missing_count
                        missing_pos_min = missing_pos
                        missing_pct = missing_count_min / len(prompt_tokens)
                prompt_feature = [missing_count_min, missing_pct]
                if len(feature_columns) > 2:
                    prompt_feature.extend(missing_pos_min)
                res['features'].extend(prompt_feature)

    def extract_ngram_features(self, remove_all=False):
        """
        Matching ngrams with sample responses
        :param remove_all: remove features
        """
        module_logger.info('------ Extract ngram features')

        self.remove_features_prefix('lem-ngram_match')

        feature_columns = ['ngram_match', 'ngram_match-lem']
        feature_columns.extend(['ngram_unseen-1', 'ngram_unseen-2', 'ngram_unseen-3'])
        self.remove_features_prefix('ngram')
        if remove_all:
            return
        self.__feature_names__.extend(feature_columns)

        for dutch_prompt in self.__student_responses__:
            sample_list = self.__reference_grammar__[dutch_prompt]['response_list']
            sample_count = len(sample_list)

            for res in self.__student_responses__[dutch_prompt]:
                response_text = res['tokens']
                response_lem = res['lemma']
                token_count = len(response_text) + 0.0
                unseen1 = token_count
                unseen2 = token_count
                unseen3 = token_count
                common123 = 0.0
                common123_lem = 0.0
                for spl in sample_list:
                    sample_response = spl['tokens']
                    sample_response_lem = spl['lemma']
                    common1 = nltk_helper.common_ngrams(response_text, sample_response, 1, False)
                    common2 = nltk_helper.common_ngrams(response_text, sample_response, 2, False)
                    common3 = nltk_helper.common_ngrams(response_text, sample_response, 3, False)
                    common123 += len(common1) + len(common2) + len(common3)

                    if unseen1 > token_count - len(common1):
                        unseen1 = token_count - len(common1)
                    if unseen2 > token_count - len(common2):
                        unseen2 = token_count - len(common2)
                    if unseen2 > token_count - len(common3):
                        unseen2 = token_count - len(common3)

                    common1 = nltk_helper.common_ngrams(response_lem, sample_response_lem, 1, False)
                    common2 = nltk_helper.common_ngrams(response_lem, sample_response_lem, 2, False)
                    common3 = nltk_helper.common_ngrams(response_lem, sample_response_lem, 3, False)
                    common123_lem += len(common1) + len(common2) + len(common3)

                ngram_feature = [common123 / sample_count, common123_lem / sample_count]
                if len(feature_columns) > 2:
                    ngram_feature.extend([unseen1 / token_count, unseen2 / token_count, unseen3 / token_count])
                res['features'].extend(ngram_feature)

    def extract_lm_features(self, ngram_file, ngram_order, lm_folder, ge_folder, remove_all=False):
        """
        Compute language model feature (probability) of a student response
        :param ngram_file: path to ngram file
        :param ngram_order: ngram order
        :param lm_folder: folder of language model files
        :param ge_folder: folder of GE file
        :param remove_all: remove features
        """
        module_logger.info('------ Extract language model features')

        feature_columns = ['ppl-ref', 'ppl-ref_pos', 'ppl-ref_prod', 'ppl-ref_dep',
                           'ppl-prompt', 'ppl-prompt_pos',
                           'ppl-correct', 'ppl-correct_pos', 'ppl-correct_prod', 'ppl-correct_dep',
                           'ppl-ge', 'ppl-ge_pos',
                           'ppl-incorrect', 'ppl-incorrect_pos', 'ppl-incorrect_prod', 'ppl-incorrect_dep']
        self.remove_features_prefix('ppl')
        if remove_all:
            return
        self.__feature_names__.extend(feature_columns)

        for dutch_prompt in self.__student_responses__:
            prompt_id = self.__reference_grammar__[dutch_prompt]['id']
            response_id_lines = []
            response_lines = []
            response_pos_lines = []
            response_prod_lines = []
            response_dep_lines = []
            for res in self.__student_responses__[dutch_prompt]:
                response_id_lines.append(res['id'])
                response_lines.append('<sos> ' + res['response'] + ' <eos>')
                response_pos_lines.append('<sos> ' + ' '.join(res['pos']) + ' <eos>')
                response_prod_lines.append(' '.join(res['prod']))
                response_dep_lines.append(' '.join(res['dep']))

            sri_lm = srilm_helper.SriLm(ngram_file=ngram_file)

            '''probability by word LM'''
            lm_file = os.path.join(lm_folder, '0.lm')
            prob = sri_lm.ngram_text(response_lines, lm_file, ngram_order)
            '''probability by POS LM'''
            lm_file = os.path.join(lm_folder, '0_pos.lm')
            prob_pos = sri_lm.ngram_text(response_pos_lines, lm_file, ngram_order)
            '''probability by production rule LM'''
            lm_file = os.path.join(lm_folder, '0_prod.lm')
            prob_prod = sri_lm.ngram_text(response_prod_lines, lm_file, 1, sos=False)
            '''probability by dependency rule LM'''
            lm_file = os.path.join(lm_folder, '0_dep.lm')
            dep_prod = sri_lm.ngram_text(response_dep_lines, lm_file, 1, sos=False)

            '''probability by word LM per prompt'''
            lm_file = os.path.join(lm_folder, prompt_id + '.lm')
            prob_prompt = sri_lm.ngram_text(response_lines, lm_file, ngram_order)
            '''probability by POS LM per prompt'''
            lm_file = os.path.join(lm_folder, prompt_id + '_pos.lm')
            pos_prompt = sri_lm.ngram_text(response_pos_lines, lm_file, ngram_order)

            '''probability from correct response LM'''
            lm_file = os.path.join(lm_folder, 'trs-words.lm')
            c_prob = sri_lm.ngram_text(response_lines, lm_file, ngram_order)
            lm_file = os.path.join(lm_folder, 'trs-pos.lm')
            c_prob_pos = sri_lm.ngram_text(response_pos_lines, lm_file, ngram_order)
            lm_file = os.path.join(lm_folder, 'trs-prod.lm')
            c_prob_prod = sri_lm.ngram_text(response_prod_lines, lm_file, 1, sos=False)
            lm_file = os.path.join(lm_folder, 'trs-prod.lm')
            c_prob_dep = sri_lm.ngram_text(response_dep_lines, lm_file, 1, sos=False)

            '''probability from incorrect response LM'''
            lm_file = os.path.join(ge_folder, 'GrammaticalErrors_lower.lm')
            e_prob_ge = sri_lm.ngram_text(response_lines, lm_file, ngram_order)
            lm_file = os.path.join(ge_folder, 'GrammaticalErrors_lower_pos.lm')
            e_prob_ge_pos = sri_lm.ngram_text(response_lines, lm_file, ngram_order)

            lm_file = os.path.join(lm_folder, 'trs-words_err.lm')
            e_prob = sri_lm.ngram_text(response_lines, lm_file, ngram_order)
            lm_file = os.path.join(lm_folder, 'trs-pos_err.lm')
            e_prob_pos = sri_lm.ngram_text(response_pos_lines, lm_file, ngram_order)
            lm_file = os.path.join(lm_folder, 'trs-prod_err.lm')
            e_prob_prod = sri_lm.ngram_text(response_prod_lines, lm_file, 1, sos=False)
            lm_file = os.path.join(lm_folder, 'trs-prod_err.lm')
            e_prob_dep = sri_lm.ngram_text(response_dep_lines, lm_file, 1, sos=False)

            lm_features = {}
            for idx, response_id in enumerate(response_id_lines):
                lm_features[response_id] = (prob[idx][1], prob_pos[idx][1], prob_prod[idx][1], dep_prod[idx][1],
                                            prob_prompt[idx][1], pos_prompt[idx][1],
                                            c_prob[idx][1], c_prob_pos[idx][1], c_prob_prod[idx][1], c_prob_dep[idx][1],
                                            e_prob_ge[idx][1], e_prob_ge_pos[idx][1],
                                            e_prob[idx][1], e_prob_pos[idx][1], e_prob_prod[idx][1], e_prob_dep[idx][1])

            for res in self.__student_responses__[dutch_prompt]:
                res['features'].extend(lm_features[res['id']])

    def extract_word2vec_features(self, sn, sg, model_folder, remove_all=False):
        """
        Similarity between student response and sample response
        :param sn: vector size
        :param sg: skip-gram or BOW
        :param model_folder: model folder
        :param remove_all: remove features
        """
        module_logger.info('------ Extract word2vec features')

        file_suffix = read_grammar.util_w2v_model_file_suffix(sn, sg)
        column_suffix = file_suffix.split('.')[0]

        feature_columns = ['maxsim' + column_suffix]
        self.remove_features(feature_columns)
        if remove_all:
            return
        self.__feature_names__.extend(feature_columns)

        w2v = gensim_helper.WordVector()

        for dutch_prompt in self.__student_responses__:
            prompt_id = self.__reference_grammar__[dutch_prompt]['id']
            w2v.load_model(os.path.join(model_folder, prompt_id + file_suffix))

            response_list = self.__reference_grammar__[dutch_prompt]['response_list']
            sample_responses = []
            for res in response_list:
                sample_responses.append(res['tokens'])

            for res in self.__student_responses__[dutch_prompt]:
                max_sim = 0.0
                response_text = res['tokens']
                if response_text[0] != '?':
                    for samp in sample_responses:
                        sim_score = w2v.sentence_similarity(response_text, samp)
                        if sim_score > max_sim:
                            max_sim = sim_score
                word2vec_feature = (max_sim,)
                res['features'].extend(word2vec_feature)

    def extract_topic_model_features(self, model_folder, remove_all=False):
        """
        Extract topic distribution of each response
        :param model_folder: folder with LDA model file
        :param remove_all: remove features
        """
        module_logger.info('------ Extract topic model features')

        feature_columns = ['lda_sim-max', 'lda_sim-min', 'lda_sim-avg']
        self.remove_features_prefix('lda')
        if remove_all:
            return
        self.__feature_names__.extend(feature_columns)

        lda_model = gensim_helper.TopicModels()
        lda_model.load_model(os.path.join(model_folder, '0.lda'), os.path.join(model_folder, '0.dic'))

        for dutch_prompt in self.__student_responses__:
            response_list = self.__reference_grammar__[dutch_prompt]['response_list']
            for res in self.__student_responses__[dutch_prompt]:
                sim_max = -1.0
                sim_min = 1.0
                sim_total = 0.0
                res_tokens = res['tokens']
                for sample_res in response_list:
                    sim = lda_model.topic_similarity(res_tokens, sample_res['tokens'])
                    if sim > sim_max:
                        sim_max = sim
                    if sim < sim_min:
                        sim_min = sim
                    sim_total += sim
                sim_features = (sim_max, sim_min, sim_total / len(response_list))
                res['features'].extend(sim_features)

    def extract_class_label(self, prediction_file=None):
        """
        Extract class label
        :param prediction_file: prediction output with logp
        """
        module_logger.info('------ Extract class label, may use prediction output ::: {}'.format(prediction_file))
        prediction_output = {}
        if prediction_file is not None and os.path.exists(prediction_file):
            pf = open(prediction_file)
            for pred in pf.readlines():
                pred_info = pred.strip().split()
                prediction_output[pred_info[0]] = (float(pred_info[1]), float(pred_info[2]))

        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                response_id = res['id']
                class_label = 0.0
                if res['language'] == 'correct':
                    class_label = 1.0
                    if response_id in prediction_output:
                        pred_info = prediction_output[response_id]
                        if pred_info[0] < 1.0 and pred_info[1] > -0.00025:
                            class_label = 0.0
                res['CLASS'] = class_label

    def remove_features(self, feature_columns):
        """
        Remove duplicate features
        :param feature_columns: list of features to remove
        """
        module_logger.info('------ Remove features ::: {}'.format(len(feature_columns)))

        removed_features = []

        fc = len(self.__feature_names__)
        for fi in range(fc - 1, -1, -1):
            if self.__feature_names__[fi] in feature_columns:
                del self.__feature_names__[fi]
                removed_features.append(fi)

        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                for fi in removed_features:
                    del res['features'][fi]

    def remove_features_prefix(self, feature_prefix):
        """
        Remove duplicate features
        :param feature_prefix: prefix in feature names
        """
        module_logger.info('------ Remove features with prefix ::: {}'.format(feature_prefix))

        removed_features = []

        fc = len(self.__feature_names__)
        for fi in range(fc - 1, -1, -1):
            if self.__feature_names__[fi].startswith(feature_prefix):
                del self.__feature_names__[fi]
                removed_features.append(fi)

        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                for fi in removed_features:
                    del res['features'][fi]

    def write_feature_sheet(self, data_file):
        """
        Write features into file
        :param data_file: data file
        """
        module_logger.info('------ Write feature sheet ::: {}'.format(data_file))

        feature_sheet = open(data_file.replace('.csv', '_features.csv'), 'wb')
        feature_sheet.write('ID\t' + '\t'.join(self.__feature_names__) + '\tCLASS\n')
        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                response_id = res['id']
                feature_row = [str(response_id)]
                try:
                    for fv in res['features']:
                        feature_row.append(str(fv))
                    feature_row.append(str(res['CLASS']))
                    feature_sheet.write('\t'.join(feature_row) + '\n')
                except:
                    module_logger.exception('****** Failed writing features for response ::: {}'.format(response_id))
        feature_sheet.close()

    def include_prediction_results(self, prediction_output, data_file, t_threshold=0.5):
        """
        Label prediction output
        :param prediction_output: raw prediction output as `accept` or `reject`
        :param data_file: name of data file
        :param t_threshold: to decide `accept` response
        """
        module_logger.info('------ Label prediction output')

        submission = codecs.open(data_file.replace('.csv', '_submission.csv'), encoding='utf-8', mode='wb')
        submission.write('\t'.join(['Id', 'Prompt', 'Wavfile', 'RecResult', 'Judgement', 'Prediction', 'Probability'])
                         + '\n')

        scores = generic_helper.init_scores()
        for dutch_prompt in self.__student_responses__:
            for res in self.__student_responses__[dutch_prompt]:
                response_id = res['id']
                language_correct_gs = res['language']
                meaning_correct_gs = res['meaning']
                pred, maxp = prediction_output[response_id]
                if pred > 0.0 or maxp < t_threshold:
                    accept_reject = 'accept'
                else:
                    accept_reject = 'reject'
                decs = generic_helper.score_decision(accept_reject, language_correct_gs, meaning_correct_gs, scores)
                res['result'] = decs
                res['accept'] = accept_reject
                a_prompt = res['prompt_list'][0]
                response_text = res['response']
                submission.write('\t'.join([response_id, a_prompt, 'n/a', response_text, accept_reject,
                                            str(pred), str(maxp)]) + '\n')

        submission.close()
        D = generic_helper.print_scores(scores, 3)
        return D


class Pipeline:

    def __init__(self):
        module_logger.info('------ Initialize the pipeline')
        self.__sr__ = StudentResponses()
        self.__rg__ = read_grammar.ReferenceGrammar()

        self.__parse_lib__ = None

        self.__ngram_count_file__ = None
        self.__ngram_file__ = None
        self.__parse_folder__ = None
        self.__lm_folder__ = None
        self.__ge_folder__ = None
        self.__edit_patterns_file__ = None

    def set_edit_patterns_file(self, edit_patterns_file):
        """
        :param edit_patterns_file:
        """
        self.__edit_patterns_file__ = edit_patterns_file

    def set_parse_lib(self, parse_lib_folder, parse_config_file):
        """
        :param parse_lib_folder: folder to parse text library
        :param parse_config_file: parse config file
        """
        module_logger.info('------ parseText library ::: %s ::: %s' % (parse_lib_folder, parse_config_file))
        self.__parse_lib__ = parse_helper.ParseText(parse_lib_folder, parse_config_file)

    def set_sri_execs(self, ngram_count_file, ngram_file):
        """
        :param ngram_count_file: path to ngram-count
        :param ngram_file: path to ngram file
        """
        module_logger.info('------ SRILM execs ::: %s ::: %s' % (ngram_count_file, ngram_file))
        self.__ngram_count_file__ = ngram_count_file
        self.__ngram_file__ = ngram_file

    def set_parse_folder(self, parse_folder):
        """
        :param parse_folder: folder to parse output
        """
        module_logger.info('------ Parse in/out folder ::: %s' % parse_folder)
        self.__parse_folder__ = parse_folder

    def set_lm_folder(self, lm_folder):
        """
        :param lm_folder: folder to language model file
        """
        module_logger.info('------ Language model in/out folder ::: %s' % lm_folder)
        self.__lm_folder__ = lm_folder

    def set_ge_folder(self, ge_folder):
        """
        :param ge_folder: folder to grammatical error corpus
        """
        module_logger.info('------ Grammatical error corpus folder ::: %s' % ge_folder)
        self.__ge_folder__ = ge_folder

    def parse_text_grammar(self, grammar_file, parsed=False, to_parse=True):
        """
        Extract responses and parse text
        :param grammar_file: reference grammar file
        :param parsed: responses were parsed
        :param to_parse: parse responses
        """
        if parsed:
            self.__rg__.json_loads(grammar_file)
        else:
            '''Read reference grammar'''
            self.__rg__.read_grammar_file(grammar_file)
            self.__rg__.response_set_to_list()
            self.__rg__.tag_prompt()
            '''Prepare input for parse text'''
            parse_text_folder = self.__rg__.write_responses_to_parse(grammar_file, self.__parse_folder__)
            if to_parse:
                self.__parse_lib__.parse(parse_text_folder)
        if parsed or to_parse:
            '''Read parse output'''
            self.__rg__.read_responses_from_parse(grammar_file, self.__parse_folder__)
        self.__rg__.json_dumps(grammar_file)

    def parse_text_data(self, data_file, asr_file, no_duplicate=False, no_trace=True, use_transcript=False,
                        parse_transcription=False, parsed=False, to_parse=True, grammar_file=None, clean_res=False):
        """
        Extract response from data file and parse text
        :param data_file: data file
        :param asr_file: ASR output
        :param no_duplicate: if True, remove duplicate responses
        :param no_trace: if True, ignore annotation trace
        :param use_transcript: if True, overwrite responses with transcripts
        :param parse_transcription: if True, extract and parse transcripts
        :param parsed: responses were parsed
        :param to_parse: parse responses
        :param grammar_file: reference grammar file
        :param clean_res: if True, do cleaning
        """
        '''Read data'''
        if parsed:
            self.__sr__.json_loads(data_file)
        else:
            self.__sr__.read_data(data_file, no_duplicate=no_duplicate, no_trace=no_trace,
                                  use_transcript=use_transcript, clean_res=clean_res)
            if asr_file is not None and os.path.isfile(asr_file):
                if grammar_file is not None and os.path.isfile(asr_file):
                    self.__sr__.read_grammar_from_json(grammar_file)
                self.__sr__.add_external_asr_nbest(asr_file, clean_asr=clean_res)
            self.__sr__.calculate_wer()
            '''Prepare input for parse text'''
            parse_text_folder = self.__sr__.write_responses_to_parse(data_file, self.__parse_folder__)
            if to_parse:
                self.__parse_lib__.parse(parse_text_folder)
            if parse_transcription:
                parse_text_folder_trans = self.__sr__.write_responses_to_parse(data_file, self.__parse_folder__, True)
                if to_parse:
                    self.__parse_lib__.parse(parse_text_folder_trans)
        if parsed or to_parse:
            '''Read parse output'''
            self.__sr__.read_responses_from_parse(data_file, self.__parse_folder__)
            if parse_transcription:
                self.__sr__.read_responses_from_parse(data_file, self.__parse_folder__, True)
        self.__sr__.json_dumps(data_file)

    def language_mode_grammar(self, grammar_file):
        """
        Train different language models for responses in reference grammar
        :param grammar_file: reference grammar XML file
        """
        self.__rg__.json_loads(grammar_file)
        '''learn word2vec embedding'''
        w2v_folder = os.path.join(self.__lm_folder__, 'word2vec')
        self.__rg__.train_word2vec_model_by_prompt(15, 1, w2v_folder)
        self.__rg__.train_word2vec_model_by_prompt(30, 1, w2v_folder)
        self.__rg__.train_word2vec_model_by_prompt(50, 1, w2v_folder)
        self.__rg__.train_word2vec_model_by_prompt(15, 0, w2v_folder)
        self.__rg__.train_word2vec_model_by_prompt(30, 0, w2v_folder)
        self.__rg__.train_word2vec_model_by_prompt(50, 0, w2v_folder)

        if not os.path.exists(self.__lm_folder__):
            os.makedirs(self.__lm_folder__)

        '''learn ngram language models'''
        sample_folder = os.path.join(self.__lm_folder__, 'samples')
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        ngram_lm_folder = os.path.join(self.__lm_folder__, 'ngrams')
        if not os.path.exists(ngram_lm_folder):
            os.makedirs(ngram_lm_folder)
        self.__rg__.train_response_lm(self.__ngram_count_file__, sample_folder, ngram_lm_folder, 5)
        '''learn LDA model'''
        lda_folder = os.path.join(self.__lm_folder__, 'lda')
        if not os.path.exists(lda_folder):
            os.makedirs(lda_folder)
        self.__rg__.train_lda_model(lda_folder, rare_threshold=2, topic_count=50)
        '''learn ngram language models of error sentences'''
        read_grammar.train_grammatical_error_lm(self.__ngram_count_file__, self.__ge_folder__, 5)

    def language_model_data(self, data_file):
        """
        Train language models for transcripts in training data
        :param data_file:
        """
        if not os.path.exists(self.__lm_folder__):
            os.makedirs(self.__lm_folder__)
        self.__sr__.json_loads(data_file)
        sample_folder = os.path.join(self.__lm_folder__, 'samples')
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        ngram_lm_folder = os.path.join(self.__lm_folder__, 'ngrams')
        if not os.path.exists(ngram_lm_folder):
            os.makedirs(ngram_lm_folder)
        self.__sr__.train_response_lm(self.__ngram_count_file__, sample_folder, ngram_lm_folder, 5, no_duplicate=False)

    def extract_feature(self, data_file, grammar_file, is_training=True):
        """
        Extract features from student responses
        :param data_file: data file
        :param grammar_file: reference grammar file
        :param is_training: if True, input data is training
        """

        self.__sr__.json_loads(data_file)
        self.__sr__.read_grammar_from_json(grammar_file)
        self.__sr__.add_translated_prompt()

        if not is_training:
            self.__sr__.read_edit_patterns(self.__edit_patterns_file__)

        ngram_lm_folder = os.path.join(self.__lm_folder__, 'ngrams')
        w2v_folder = os.path.join(self.__lm_folder__, 'word2vec')
        lda_folder = os.path.join(self.__lm_folder__, 'lda')
        '''extract features'''
        self.__sr__.extract_topic_model_features(lda_folder)
        self.__sr__.extract_word2vec_features(15, 1, w2v_folder)
        self.__sr__.extract_word2vec_features(30, 1, w2v_folder)
        self.__sr__.extract_word2vec_features(50, 1, w2v_folder)
        self.__sr__.extract_word2vec_features(15, 0, w2v_folder)
        self.__sr__.extract_word2vec_features(30, 0, w2v_folder)
        self.__sr__.extract_word2vec_features(50, 0, w2v_folder)
        self.__sr__.extract_lm_features(self.__ngram_file__, 5, ngram_lm_folder, self.__ge_folder__)
        self.__sr__.extract_ngram_features()
        self.__sr__.extract_spelling_features()
        self.__sr__.extract_parse_score_features()
        self.__sr__.extract_length_features()
        self.__sr__.extract_prompt_features()
        self.__sr__.extract_edit_features(reset_edit_list=is_training)
        '''extract class label and save feature sheet'''
        # self.__sr__.remove_features(to_remove)
        self.__sr__.extract_class_label()
        self.__sr__.write_feature_sheet(data_file)
        self.__sr__.json_dumps(data_file)

        if is_training:
            self.__sr__.write_edit_patterns(self.__edit_patterns_file__)

        self.ml(data_file)

    def combined_sets(self, data_file_output, data_files):
        """
        :param data_file_output: output data file name
        :param data_files: list of extracted sheets
        """
        print '[INFO]', data_file_output
        self.__sr__.json_loads(data_files[0])
        if len(data_files) > 1:
            for i in range(1, len(data_files)):
                self.__sr__.json_loads_more(data_files[i])
        self.__sr__.write_feature_sheet(data_file_output)
        self.__sr__.json_dumps(data_file_output)

    def ml(self, data_file, test_file=None):
        """
        Run prediction
        :param data_file: training data file
        :param test_file: test data file
        """
        feature_sheet = data_file.replace('.csv', '_features.csv')
        removed_columns = ml.feature_index(feature_sheet, feature_names.to_remove, feature_names.to_keep)
        if test_file is None:
            self.__sr__.json_loads(data_file)
            prediction_output = ml.xval(feature_sheet, removed_columns)
            D = self.__sr__.include_prediction_results(prediction_output, data_file)
        else:
            test_sheet = test_file.replace('.csv', '_features.csv')
            self.__sr__.json_loads(test_file)
            prediction_output = ml.train_test(feature_sheet, test_sheet, removed_columns)
            D = self.__sr__.include_prediction_results(prediction_output, test_file)
        return D

    def ml_fx(self, data_file, test_file=None):
        """
        Run prediction
        :param data_file: training data file
        :param test_file: test data file
        """
        self.__sr__.json_loads(data_file)
        all_edits = [fn for fn in self.__sr__.__feature_names__ if fn.startswith('edit_pattern')]

        feature_names_to_keep = list(feature_names.to_keep)
        feature_names_to_remove = list(feature_names.to_remove)

        feature_names.to_keep = []
        D = self.ml(data_file, test_file)
        module_logger.info('++++++ no edit-pattern ::: {}'.format(D))

        res = dict()
        for fn in all_edits:
            feature_names.to_keep = [fn]
            Da = self.ml(data_file, test_file)
            if Da > D:
                res[fn] = Da
                module_logger.info('++++++ edit-pattern ::: {} ::: {}'.format(fn, Da))

        feature_names.to_keep = list(res.keys())
        D = self.ml(data_file, test_file)
        module_logger.info('++++++ selected edit-pattern ::: {}'.format(D))

        for fn in feature_names.to_keep:
            feature_names.to_remove = feature_names_to_remove + [fn]
            Da = self.ml(data_file, test_file)
            if Da > D:
                del res[fn]
                module_logger.info('++++++ edit-pattern removed ::: {} ::: {}'.format(fn, Da))

        feature_names.to_keep = list(res.keys())
        D = self.ml(data_file, test_file)
        module_logger.info('++++++ not-removed edit-pattern remaining ::: {}'.format(D))
        for fn in sorted(res.keys()):
            print '[----] edit pattern', fn, res[fn]

        feature_names.to_keep = feature_names_to_keep
        feature_names.to_remove = feature_names_to_remove

        with open(data_file.replace('.csv', '_top-edits.json')) as jf:
            json.dump(res.keys(), jf)

    def write_prompt_distribution(self, data_file, grammar_file):
        """
        Extract features from student responses
        :param data_file: data file
        :param grammar_file: reference grammar file
        """

        self.__sr__.json_loads(data_file)
        self.__sr__.read_grammar_from_json(grammar_file)
        self.__sr__.add_translated_prompt()
        self.__sr__.get_prompt_distribution(data_file)


RUN_MODES = ['parse', 'lm', 'extract', 'ml', 'fx', 'trs', 'cmb', 'pdf', 'wer']
DATA_MODES = ['grammar', 'training', 'test', 'all']
PARSED = False
TO_PARSE = False
RUN_MODE = RUN_MODES[8]
DATA_MODE = DATA_MODES[2]

ASR_TRAINING_FILE_BASE = ''
ASR_TEST_FILE_BASE = ''

'''2017 data'''
# __SPOKENCALL__ = '/Users/huynguyen/Downloads/spokenCALL'
__SPOKENCALL__ = 'c:/Users/hnguyen6/Downloads/spokenCALL'

# ROOT_FOLDER = os.path.join(__SPOKENCALL__, '2017_v3')
# GRAMMAR_FILE_BASE = 'referenceGrammar.xml'
# TRAINING_FILE_BASE = 'textProcessing_trainingKaldi.csv'
# EDIT_FILE_BASE = TRAINING_FILE_BASE.replace('.csv', '_edit.json')
# ASR_TRAINING_FILE_BASE = 'nbest5-19-1.txt'

'''2018 data'''
ROOT_FOLDER = os.path.join(__SPOKENCALL__, '2018_v8_WER')
GRAMMAR_FILE_BASE = 'referenceGrammar.xml'
TRAINING_FILE_BASE = 'scst2_training_data_1ABC_text.csv'
EDIT_FILE_BASE = TRAINING_FILE_BASE.replace('.csv', '_edit.json')

# TRAINING_FILE_BASE = 'scst2_training_data_17ABC_text.csv'
# ASR_TRAINING_FILE_BASE = '2018-nbest-files/nbest1-19-1.csv'
# TRAINING_FILE_BASE = 'textProcessing_trainingKaldi.csv'

DATA_FILE_COMBINED_BASE = 'scst2_training_data_17ABC_text.csv'
DATA_FILES_BASE = [
    'textProcessing_trainingKaldi.csv',
    'scst2_training_data_A_text.csv',
    'scst2_training_data_B_text.csv',
    'scst2_training_data_C_text.csv',
]

TEST_FILE_BASE = 'scst2_training_data_ABC_text.csv'
# TEST_FILE_BASE = 'scst2_training_data_B_text.csv'
# TEST_FILE_BASE = 'scst2_training_data_C_text.csv'
ASR_TEST_FILE_BASE = '2018-nbest-files/nbest1-19-1.csv'

# TEST_FILE_BASE = 'textProcessing_trainingKaldi.csv'
# ASR_TEST_FILE_BASE = '2017-nbest-files/nbest1-19-1.csv'
# TEST_FILE_BASE = 'textProcessing_testKaldi_annotated.csv'
# ASR_TEST_FILE_BASE = '2017-nbest-files-test/nbest.2017-testset-withlma-1best-21-0.0.hyp'

# TEST_FILE_BASE = 'scst2_testDataText.csv'
# ASR_TEST_FILE_BASE = '2018-nbest-files-test/nbest.2018-testset-withlma-5best-21-0.0.hyp'

# TRAINING_FILE_BASE = TEST_FILE_BASE
# TEST_FILE_BASE = TRAINING_FILE_BASE

GE_FOLDER = os.path.join(ROOT_FOLDER, 'ge')
PARSE_FOLDER = os.path.join(ROOT_FOLDER, 'parse')
LM_FOLDER = os.path.join(ROOT_FOLDER, 'lm')

GRAMMAR_FILE = os.path.join(ROOT_FOLDER, 'data', GRAMMAR_FILE_BASE)
TRAINING_FILE = os.path.join(ROOT_FOLDER, 'data', TRAINING_FILE_BASE)
DATA_FILE_COMBINED = os.path.join(ROOT_FOLDER, 'data', DATA_FILE_COMBINED_BASE)
DATA_FILES = [os.path.join(ROOT_FOLDER, 'data', tfb) for tfb in DATA_FILES_BASE]
TEST_FILE = os.path.join(ROOT_FOLDER, 'data', TEST_FILE_BASE)

ASR_TRAINING_FILE = os.path.join(ROOT_FOLDER, 'data', ASR_TRAINING_FILE_BASE)
ASR_TEST_FILE = os.path.join(ROOT_FOLDER, 'data', ASR_TEST_FILE_BASE)

EDIT_FILE = os.path.join(ROOT_FOLDER, 'data', EDIT_FILE_BASE)

'''Lazy variables providing default values to command-line arguments'''
# SRI_FOLDER = '/Users/huynguyen/Downloads/srilm-1.7.2/bin/macosx'
SRI_FOLDER = 'c:/Users/hnguyen6/Downloads/srilm-1.7.2/bin/x64-Release'
SRI_NGRAM = os.path.join(SRI_FOLDER, 'ngram.exe')
SRI_NGRAM_COUNT = os.path.join(SRI_FOLDER, 'ngram-count.exe')
# PARSE_LIB_FOLDER = '/Users/huynguyen/Downloads/ArgminingDemo/parseText_lib'
# PARSE_CONFIG_FILE = '/Users/huynguyen/Downloads/ArgminingDemo/parseText-lines.config'
PARSE_LIB_FOLDER = 'c:/Users/hnguyen6/Downloads/parseText_lib'
PARSE_CONFIG_FILE = 'c:/Users/hnguyen6/Dropbox/workspace/CODES/parseText/parseText-lines.config'


def rume():

    time_begin = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--sri-ngram', action='store', dest='sri_ngram',
                        help='Path to SRILM ngram file', default=SRI_NGRAM, type=str)
    parser.add_argument('--sri-ngram-count', action='store', dest='sri_ngram_count',
                        help='Path to SRILM ngram-count file', default=SRI_NGRAM_COUNT, type=str)
    parser.add_argument('--parse-lib-folder', action='store', dest='parse_lib_folder',
                        help='Path to parseText jar files', default=PARSE_LIB_FOLDER, type=str)
    parser.add_argument('--parse-config-file', action='store', dest='parse_config_file',
                        help='Path to parseText config file', default=PARSE_CONFIG_FILE, type=str)
    parser.add_argument('--lm-folder', action='store', dest='lm_folder',
                        help='Path to language model input/output files', default=LM_FOLDER, type=str)
    parser.add_argument('--ge-folder', action='store', dest='ge_folder',
                        help='Path to grammatical error corpus', default=GE_FOLDER, type=str)
    parser.add_argument('--parse-folder', action='store', dest='parse_folder',
                        help='Path to parse in/out folder', default=PARSE_FOLDER, type=str)
    parser.add_argument('--grammar-file', action='store', dest='grammar_file',
                        help='Path to reference grammar XML file', default=GRAMMAR_FILE, type=str)
    parser.add_argument('--data-file', action='store', dest='data_file',
                        help='Path to training data file. Required in ML mode', default=TRAINING_FILE, type=str)
    parser.add_argument('--test-file', action='store', dest='test_file',
                        help='Path to test data file', default=TEST_FILE, type=str)
    parser.add_argument('--asr-file', action='store', dest='asr_data_file',
                        help='Path to ASR file for training data', default=ASR_TRAINING_FILE, type=str)
    parser.add_argument('--asr-test-file', action='store', dest='asr_test_file',
                        help='Path to ASR file for test data', default=ASR_TEST_FILE, type=str)
    parser.add_argument('--edit-patterns-file', action='store', dest='edit_patterns_file',
                        help='File of edits found in training data', default=EDIT_FILE, type=str)
    parser.add_argument('--run-mode', action='store', dest='run_mode',
                        help='Run modes: parse|lm|extract|ml|fx|trs', default=RUN_MODE, type=str)

    arguments = parser.parse_args()

    pipe = Pipeline()
    pipe.set_sri_execs(arguments.sri_ngram_count, arguments.sri_ngram)
    pipe.set_parse_lib(arguments.parse_lib_folder, arguments.parse_config_file)
    pipe.set_ge_folder(arguments.ge_folder)
    pipe.set_lm_folder(arguments.lm_folder)
    pipe.set_parse_folder(arguments.parse_folder)
    pipe.set_edit_patterns_file(arguments.edit_patterns_file)

    run_mode = arguments.run_mode
    module_logger.info('------ Run mode ::: %s' % run_mode)

    if run_mode == 'parse' or run_mode == 'trs' or run_mode == 'wer':
        if DATA_MODE == 'grammar':
            grammar_file = arguments.grammar_file
            if os.path.exists(grammar_file):
                pipe.parse_text_grammar(grammar_file, parsed=PARSED, to_parse=TO_PARSE)
            else:
                module_logger.error('****** Grammar file not found/provided ::: {}'.format(grammar_file))
        elif DATA_MODE == 'training':
            data_file = arguments.data_file
            asr_data_file = arguments.asr_data_file
            if os.path.exists(data_file):
                pipe.parse_text_data(data_file, asr_file=asr_data_file, parsed=PARSED, to_parse=TO_PARSE,
                                     parse_transcription=True, no_trace=False, use_transcript=run_mode == 'trs',
                                     clean_res=run_mode != 'wer', grammar_file=arguments.grammar_file)
            else:
                module_logger.error('****** Data file not found/provided ::: {}'.format(data_file))
        elif DATA_MODE == 'test':
            test_file = arguments.test_file
            asr_test_file = arguments.asr_test_file
            if os.path.exists(test_file):
                pipe.parse_text_data(test_file, asr_file=asr_test_file, parsed=PARSED, to_parse=TO_PARSE,
                                     parse_transcription=False, no_trace=True, use_transcript=run_mode == 'trs',
                                     clean_res=run_mode != 'wer', grammar_file=arguments.grammar_file)
            else:
                module_logger.error('****** Test file not found/provided ::: {}'.format(test_file))

    elif run_mode == 'lm':
        if DATA_MODE == 'grammar':
            grammar_file = arguments.grammar_file
            if os.path.exists(grammar_file):
                pipe.language_mode_grammar(grammar_file)
            else:
                module_logger.error('****** Grammar file not found/provided ::: {}'.format(grammar_file))
        elif DATA_MODE == 'training':
            data_file = arguments.data_file
            if os.path.exists(data_file):
                pipe.language_model_data(data_file)
            else:
                module_logger.error('****** Data file not found/provided ::: {}'.format(data_file))

    elif run_mode == 'extract':
        grammar_file = arguments.grammar_file
        if os.path.exists(grammar_file):
            if DATA_MODE == 'training':
                data_file = arguments.data_file
                if os.path.exists(data_file):
                    pipe.extract_feature(data_file, grammar_file, True)
                else:
                    module_logger.error('****** Data file not found/provided ::: {}'.format(data_file))
            elif DATA_MODE == 'test':
                test_file = arguments.test_file
                if os.path.exists(test_file):
                    pipe.extract_feature(test_file, grammar_file, False)
                else:
                    module_logger.error('****** Test file not found/provided ::: {}'.format(test_file))

    elif run_mode == 'ml':
        data_file = arguments.data_file
        if DATA_MODE == 'training':
            test_file = ''
        else:
            test_file = arguments.test_file
        if os.path.exists(data_file.replace('.csv', '.json')):
            if not os.path.exists(test_file):
                test_file = None
            pipe.ml(data_file, test_file)

    elif run_mode == 'fx':
        data_file = arguments.data_file
        if DATA_MODE == 'training':
            test_file = ''
        else:
            test_file = arguments.test_file
        if os.path.exists(data_file.replace('.csv', '.json')):
            if not os.path.exists(test_file):
                test_file = None
            pipe.ml_fx(data_file, test_file)

    elif run_mode == 'cmb':
        if DATA_MODE == 'all':
            pipe.combined_sets(DATA_FILE_COMBINED, DATA_FILES)

    elif run_mode == 'pdf':
        grammar_file = arguments.grammar_file
        data_file = None
        if DATA_MODE == 'training':
            data_file = arguments.data_file
        elif DATA_MODE == 'test':
            data_file = arguments.test_file
        if data_file is not None and os.path.exists(data_file):
            pipe.write_prompt_distribution(data_file, grammar_file)
        else:
            module_logger.error('****** Data file not found/provided ::: {}'.format(data_file))

    run_time = time.time() - time_begin
    print '\n[!!!!] data transformation done', run_time


if __name__ == '__main__':
    rume()
