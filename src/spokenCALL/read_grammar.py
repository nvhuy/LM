"""
Created on Dec 28, 2017

@author: HuyNguyen
"""
import os
import codecs
import logging
import json
import re
module_logger = logging.getLogger(__name__)

project_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xml.etree.ElementTree as Et

from src.utils import srilm_helper, nltk_helper, gensim_helper, generic_helper


class ReferenceGrammar:

    def __init__(self):
        """a dictionary {prompt : {id, translation, {responses}}}"""
        self.__sample_responses__ = {}
        self.__max_id__ = 1
        self.__verb_contraction_list__ = generic_helper.read_verb_long_short_list(os.path.join(project_folder,
                                                                                               'resource',
                                                                                               'verb-contracted.txt'))
        self.__head_pattern_list__ = generic_helper.create_head_word_patterns()

    def read_grammar_file(self, grammar_file):
        """
        Read grammar file and store data in a dictionary
        {prompt : {id, translation, responses}}
        :param grammar_file: grammar file
        :return: a dictionary
        """
        module_logger.info('------ Parse reference grammar file ::: {}'.format(grammar_file))

        try:
            del self.__sample_responses__
        except:
            module_logger.exception('****** Sample response variable not initialized before')
        finally:
            self.__sample_responses__ = dict()

        prompt_file = codecs.open(grammar_file.replace('.xml', '.prm'), mode='wb', encoding='utf-8')

        prompt_id = 0
        duplicate_prompts = 0
        try:
            xmlt = Et.parse(grammar_file).getroot()
            for chld in xmlt:
                if chld.tag == 'prompt_unit':
                    prompt_id += 1
                    dutch_prompt = None
                    translated_prompt = None
                    response_set = []
                    for grand_child in chld:
                        if grand_child.tag == 'prompt':
                            dutch_prompt = grand_child.text
                        elif grand_child.tag == 'translated_prompt':
                            translated_prompt = grand_child.text
                            if ':' in translated_prompt:
                                translated_prompt = translated_prompt.rsplit(':')[1].strip()
                        elif grand_child.tag == 'response':
                            response_text = grand_child.text
                            response_text = generic_helper.remove_head_words_response(response_text,
                                                                                      self.__head_pattern_list__)
                            new_responses = generic_helper.replace_contracted_verb(response_text,
                                                                                   self.__verb_contraction_list__)
                            response_set.extend(new_responses)
                    if len(response_set) > 0 and translated_prompt is not None and dutch_prompt is not None:
                        if dutch_prompt not in self.__sample_responses__:
                            prompt_file.write(dutch_prompt + '\n')
                            self.__sample_responses__[dutch_prompt] = {}
                            self.__sample_responses__[dutch_prompt]['id'] = str(prompt_id)
                            self.__sample_responses__[dutch_prompt]['translated_prompt'] = []
                            self.__sample_responses__[dutch_prompt]['response_set'] = []
                        else:
                            duplicate_prompts += 1
                        self.__sample_responses__[dutch_prompt]['translated_prompt'].append(translated_prompt)
                        self.__sample_responses__[dutch_prompt]['response_set'].extend(response_set)
            self.__max_id__ = prompt_id
            module_logger.warning('****** Duplicate prompts ::: {}'.format(duplicate_prompts))
            prompt_file.close()
        except:
            module_logger.exception('****** File was not parsed into XML tree ::: {}'.format(grammar_file))
            self.__sample_responses__ = None

    def add_correct_response(self, correct_responses):
        """
        Include correct responses from training data
        :param correct_responses: correct responses as a dictionary {prompt : [response]}
        """
        module_logger.info('------ Add correct responses from training data')

        new_responses_added = 0
        new_prompts_added = 0
        for dutch_prompt in correct_responses:
            if dutch_prompt not in self.__sample_responses__:
                # module_logger.info('------ Add new prompt ::: ' + dutch_prompt)
                new_prompts_added += 1
                self.__max_id__ += 1
                prompt_id = self.__max_id__
                self.__sample_responses__[dutch_prompt]['id'] = str(prompt_id)
                self.__sample_responses__[dutch_prompt]['translated_prompt'] = [correct_responses[dutch_prompt][0]]
                self.__sample_responses__[dutch_prompt]['response_set'] = []

            current_res_count = len(self.__sample_responses__[dutch_prompt]['response_set'])
            for correct_response in correct_responses[dutch_prompt]:
                new_responses_added = generic_helper.replace_contracted_verb(correct_response,
                                                                             self.__verb_contraction_list__)
                self.__sample_responses__[dutch_prompt]['response_set'].extend(new_responses_added)
            new_responses_added += len(self.__sample_responses__[dutch_prompt]['response_set']) - current_res_count
        module_logger.info('------ Added new prompts ::: {}'.format(new_prompts_added))
        module_logger.info('------ Added new responses ::: {}'.format(new_responses_added))

    def response_set_to_list(self):
        """
        Create a list of response from response set. Each response in list can have attributes
        """
        module_logger.info('------ Create response list')

        for dutch_prompt in self.__sample_responses__:
            response_list = []
            response_set = self.__sample_responses__[dutch_prompt]['response_set']
            prompt_id = self.__sample_responses__[dutch_prompt]['id']
            rid = 1
            for res in response_set:
                response_id = prompt_id + ':' + str(rid)
                response_list.append({'text': res, 'id': response_id})
                rid += 1
            self.__sample_responses__[dutch_prompt]['response_list'] = response_list
            del self.__sample_responses__[dutch_prompt]['response_set']

    def tag_responses(self):
        """
        Tag sample responses
        """
        module_logger.info('------ Tag sample responses')
        if self.__sample_responses__ is None:
            module_logger.error('****** Sample response is Null')
            return

        for dutch_prompt in self.__sample_responses__:
            response_list = self.__sample_responses__[dutch_prompt]['response_list']
            for res in response_list:
                response_text = res['text']
                response_tokens = nltk_helper.tokenize_sentence(response_text)
                res['tokens'] = response_tokens
                response_pos = nltk_helper.tag_tokens(response_tokens)
                res['pos'] = response_pos

    def tag_prompt(self):
        """
        Tag sample responses
        """
        module_logger.info('------ Tag prompts')
        if self.__sample_responses__ is None:
            module_logger.error('****** Sample response is Null')
            return

        for dutch_prompt in self.__sample_responses__:
            trans_expanded = []
            for translated_prompt in self.__sample_responses__[dutch_prompt]['translated_prompt']:
                trans_tokens = nltk_helper.tokenize_sentence(translated_prompt)
                trans_pos = nltk_helper.tag_tokens(trans_tokens)
                '''remove punc in translation prompt'''
                remove_id = []
                for pi in range(len(trans_pos) - 1, -1, -1):
                    if trans_pos[pi] == '.':
                        remove_id.append(pi)
                for pi in remove_id:
                    del trans_tokens[pi]
                    del trans_pos[pi]
                trans_expanded.extend(generic_helper.expand_string(trans_tokens))

            trans_expanded_str = set()
            for trans in trans_expanded:
                trans_expanded_str.add(' '.join(trans))

            prompt_list = []
            for trx in trans_expanded_str:
                pr_dict = dict()
                pr_dict['tokens'] = nltk_helper.tokenize_sentence(trx)
                pr_dict['pos'] = nltk_helper.tag_tokens(pr_dict['tokens'])
                prompt_list.append(pr_dict)

                if len(trx) == 0:
                    print '[ERRR] empty prompt due to wrong expansion'

            self.__sample_responses__[dutch_prompt]['prompt_list'] = prompt_list

    def write_responses_to_parse(self, grammar_file, parse_folder):
        """
        :param grammar_file: grammar file
        :param parse_folder: folder with input files for parser
        :return: input folder for parse text
        """
        module_logger.info('------ Write response to parse')

        grammar_file_base = os.path.splitext(os.path.basename(grammar_file))[0]
        parse_text_folder = os.path.join(parse_folder, grammar_file_base)
        if not os.path.exists(parse_text_folder):
            os.makedirs(parse_text_folder)

        r_file = codecs.open(os.path.join(parse_text_folder, grammar_file_base + '.res'), mode='wb', encoding='utf-8')
        rid_file = codecs.open(os.path.join(parse_folder, grammar_file_base + '.id'), mode='wb', encoding='utf-8')

        for dutch_prompt in self.__sample_responses__:
            response_list = self.__sample_responses__[dutch_prompt]['response_list']
            for res in response_list:
                response_text = res['text']
                response_id = res['id']
                r_file.write(response_text + '\n')
                rid_file.write(response_id + '\n')
        r_file.close()
        rid_file.close()
        return parse_text_folder

    def read_responses_from_parse(self, grammar_file, parse_folder):
        """
        Read parse output of responses
        :param grammar_file: grammar file
        :param parse_folder: folder of parsing output
        """
        module_logger.info('------ Read parse of responses')
        if self.__sample_responses__ is None:
            module_logger.error('****** Sample response is Null')
            return

        grammar_file_name = os.path.basename(grammar_file)
        parse_file_base = os.path.join(parse_folder, grammar_file_name)

        id_file = parse_file_base.replace('.xml', '.id')
        parse_file = parse_file_base.replace('.xml', '.parsed')
        response_parsed_dict = generic_helper.read_responses_from_parse(parse_file, id_file)
        if response_parsed_dict is None:
            return

        for dutch_prompt in self.__sample_responses__:
            response_list = self.__sample_responses__[dutch_prompt]['response_list']
            for res in response_list:
                response_id = res['id']
                response_parsed = response_parsed_dict[response_id]
                for parse_feat in response_parsed:
                    res[parse_feat] = response_parsed[parse_feat]
                    if parse_feat == 'parse':
                        res['prod'] = nltk_helper.extract_production_rules(res['parse'], include_leaf=False)
                    if parse_feat == 'dependency':
                        res['dep'] = generic_helper.reduce_dependency_rules(res['dependency'])

    def collect_sample_responses(self):
        """
        Return a dictionary of prompt_id, response list
        """
        module_logger.info('------ Collect sample responses')
        if self.__sample_responses__ is None:
            module_logger.error('****** Sample response is Null')
            return None
        sample_responses = {}
        for dutch_prompt in self.__sample_responses__:
            prompt_id = self.__sample_responses__[dutch_prompt]['id']
            response_list = self.__sample_responses__[dutch_prompt]['response_list']
            sample_responses[prompt_id] = []
            for res in response_list:
                sample_responses[prompt_id].append(res['tokens'])
        return sample_responses

    def write_prompt_ids(self, prompt_id_file):
        """
        Save mapping prompt -- id to file
        :param prompt_id_file: file to keep prompt ids
        """
        module_logger.info('------ Write prompt ids to files ::: {}'.format(prompt_id_file))
        if self.__sample_responses__ is None:
            module_logger.error('****** Sample response is Null')
            return

        prompt_id_mapping = {}
        for dutch_prompt in self.__sample_responses__:
            prompt_id = self.__sample_responses__[dutch_prompt]['id']
            prompt_id_mapping[dutch_prompt] = prompt_id
        try:
            jfile = open(prompt_id_file, mode='wb')
            json.dump(prompt_id_mapping, jfile)
            jfile.close()
        except:
            module_logger.exception('****** Failed dumping prompt id mapping to JSON file')

    def json_dumps(self, grammar_file):
        """
        Dump sample responses to JSON file
        :param grammar_file: reference grammar XML file
        :return: name of JSON file
        """
        module_logger.info('------ Dump sample responses to JSON file ::: {}'.format(grammar_file))
        if self.__sample_responses__ is None:
            module_logger.error('****** Sample response is Null')
            return None

        grammar_file_json = grammar_file.replace('.xml', '.json')
        try:
            jfile = open(grammar_file_json, mode='wb')
            json.dump(self.__sample_responses__, jfile, indent=2, sort_keys=True)
            jfile.close()
            print '[INFO] prompt info'
            for dutch_prompt in self.__sample_responses__:
                print ' *** '.join(sorted(self.__sample_responses__[dutch_prompt]))
                break
        except:
            module_logger.exception('****** Failed saving sample responses to JSON file')
            return None

        return grammar_file_json

    def json_loads(self, grammar_file):
        """
        Load sample responses from JSON file
        """
        module_logger.info('------ Load sample responses from JSON file ::: {}'.format(grammar_file))

        try:
            del self.__sample_responses__
            jfile = open(grammar_file.replace('.xml', '.json'), mode='rb')
            self.__sample_responses__ = json.load(jfile)
            jfile.close()
        except:
            module_logger.exception('****** Failed loading sample responses from JSON file')
            self.__sample_responses__ = None

    def train_response_lm(self, ngram_count_file, sample_folder, lm_folder, ngram_order):
        """
        Save sample responses of each prompt to a file and train language models
        :param ngram_count_file: path to ngram-count
        :param sample_folder: data output folder path
        :param lm_folder: folder path to keep LM files
        :param ngram_order: ngram order
        """
        module_logger.info('------ Train language models of sample responses::: {}'.format(sample_folder))
        if self.__sample_responses__ is None:
            module_logger.error('****** Sample response is Null')
            return

        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

        wrd_file = codecs.open(os.path.join(sample_folder, '0'), mode='wb', encoding='utf-8')
        pos_file = codecs.open(os.path.join(sample_folder, '0_pos'), mode='wb', encoding='utf-8')
        prd_file = codecs.open(os.path.join(sample_folder, '0_prod'), mode='wb', encoding='utf-8')
        dep_file = codecs.open(os.path.join(sample_folder, '0_dep'), mode='wb', encoding='utf-8')

        for dutch_prompt in self.__sample_responses__:
            try:
                prompt_id = self.__sample_responses__[dutch_prompt]['id']
                response_list = self.__sample_responses__[dutch_prompt]['response_list']

                wrd_prompt_file = codecs.open(os.path.join(sample_folder, prompt_id),
                                              mode='wb', encoding='utf-8')
                pos_prompt_file = codecs.open(os.path.join(sample_folder, prompt_id + '_pos'),
                                              mode='wb', encoding='utf-8')
                for res in response_list:
                    wrd = ' '.join(res['tokens'])
                    pos = ' '.join(res['pos'])
                    prod = ' '.join(res['prod'])
                    dep = ' '.join(res['dep'])

                    wrd_file.write(wrd + '\n')
                    pos_file.write(pos + '\n')
                    prd_file.write(prod + '\n')
                    dep_file.write(dep + '\n')

                    wrd_prompt_file.write(wrd + '\n')
                    pos_prompt_file.write(pos + '\n')

                wrd_prompt_file.close()
                pos_prompt_file.close()

            except:
                module_logger.exception('****** Failed saving sample prompt')

        wrd_file.close()
        pos_file.close()
        prd_file.close()
        dep_file.close()

        sri_lm = srilm_helper.SriLm(ngram_count_file=ngram_count_file)
        '''train word models'''
        name_pattern = re.compile('^\d+$')
        sri_lm.train_language_models(sample_folder, name_pattern, lm_folder, ngram_order)
        '''train POS models'''
        name_pattern = re.compile('^\d+_pos$')
        sri_lm.train_language_models(sample_folder, name_pattern, lm_folder, ngram_order)
        '''train production rule models'''
        name_pattern = re.compile('^\d+_prod$')
        sri_lm.train_language_models(sample_folder, name_pattern, lm_folder, 1, sos=False)
        '''train dependency rules models'''
        name_pattern = re.compile('^\d+_dep$')
        sri_lm.train_language_models(sample_folder, name_pattern, lm_folder, 1, sos=False)

    def train_word2vec_model_by_prompt(self, sn, sg, model_folder):
        """
        Train word2vec model for responses of each prompt
        :param sn: dimension of the feature vector
        :param sg: 1 to use skip ngram, 0 to use CBW
        :param model_folder: folder to save language model files
        """
        module_logger.info('------ Train word2vec model by prompt to files ::: {}'.format(model_folder))

        if self.__sample_responses__ is None:
            module_logger.error('****** Sample response is Null')
            return
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        w2v = gensim_helper.WordVector(dim=sn, skip_gram=sg, neg_samlping=10)
        file_suffix = util_w2v_model_file_suffix(sn, sg)

        for dutch_prompt in self.__sample_responses__:
            prompt_id = self.__sample_responses__[dutch_prompt]['id']
            response_list = self.__sample_responses__[dutch_prompt]['response_list']

            sentence_list = []
            for res in response_list:
                sentence_list.append(res['tokens'])

            model_file = os.path.join(model_folder, prompt_id + file_suffix)
            w2v.train_save_model(sentence_list, model_file)

    def train_lda_model(self, model_folder, rare_threshold=2, topic_count=50):
        """
        Train and save topic model from sample responses
        :param rare_threshold: remove words with occurrence less than threshold
        :param topic_count: number of topics
        :param model_folder: folder to save LDA model file
        """
        module_logger.info('------ Train LDA topic model ::: {} ::: {}'.format(rare_threshold, topic_count))

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        documents = []
        for dutch_prompt in self.__sample_responses__:
            prompt_doc = []
            response_list = self.__sample_responses__[dutch_prompt]['response_list']
            for res in response_list:
                prompt_doc.extend(res['tokens'])
            documents.append(prompt_doc)

        lda_model = gensim_helper.TopicModels()
        lda_model.train_model(documents, rare_threshold=rare_threshold, topic_count=topic_count)
        lda_model.save_model(os.path.join(model_folder, '0.lda'), os.path.join(model_folder, '0.dic'))


def util_w2v_model_file_suffix(vector_size, skip_gram):
    """
    Word2vec model file name suffix
    """
    suffix = '_' + str(vector_size)
    if skip_gram == 0:
        suffix += '_cbw'
    else:
        suffix += '_skip'
    suffix += '.w2v'
    return suffix


def train_grammatical_error_lm(sri_folder, ge_folder, ngram_order):
    """
    Train language models for dataset of error sentences
    :param sri_folder: folder of SRI lm toolkits
    :param ge_folder: folder path to keep dataset
    :param ngram_order: ngram order
    """
    sri_lm = srilm_helper.SriLm(sri_folder)
    '''train incorrect English models'''
    name_pattern = re.compile('^GrammaticalErrors_lower$')
    sri_lm.train_language_models(ge_folder, name_pattern, ge_folder, ngram_order)
    name_pattern = re.compile('^GrammaticalErrors_lower_pos$')
    sri_lm.train_language_models(ge_folder, name_pattern, ge_folder, ngram_order)
