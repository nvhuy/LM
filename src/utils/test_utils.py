"""
Created on Dec 27, 2017

@author: HuyNguyen
"""
import os
import logging

from src.spokenCALL import ml

module_logger = logging.getLogger(__name__)

import log_helper

project_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_folder = os.path.join(project_folder, 'logs')
_, GLOBAL_rotate_handler = log_helper.setup_root_logger('utils', date_only=True, unique_id=False,
                                                        console_logging_level=logging.INFO,
                                                        log_file_directory=log_folder)

from src.utils import xml_helper, nltk_helper, spell_helper, generic_helper, gensim_helper


def test_xml_helper():
    """
    Test xml_helper module
    """
    xml_file = 'd:/spokenCall/2017/data/referenceGrammar.xml'
    xh = xml_helper.xmlHelper()
    xh.parse_xml_file(xml_file)


def test_nltk():
    """
    Test nltk_helper module
    """
    raw_text = "(i would like to pay by ?credit ( card | cards ) | i would like to pay with ( ?a ?credit card | ?credit cards ))"
    tokens = nltk_helper.tokenize_sentence(raw_text)
    print generic_helper.expand_string(tokens)


def test_spell_helper():
    """
    Test spell_helper module
    """
    spch = spell_helper.EnchantSpell()
    print spch.check_tokens(['um', 'uhkkkk'])
    print spch.suggest_correction('helo')


def test_generic_helper():
    """
    test generic_helper module
    """
    # filler_pattern_list = generic_helper.create_filler_sound_patterns()
    # response = 'can i have a room there a room for the next six nights'
    # response = generic_helper.remove_filler_sounds(response, filler_pattern_list)
    # response = generic_helper.remove_word_repetition(response)
    # print response

    print generic_helper.distance_sentences(['a', 'b', 'c', 'd'], ['a', 'a', 'a', 'a', 'b', 'd'])


def test_edit_distance():
    sents = [
        ["are", "visa", "accepted"],
        ["are", "visa", "okay"],
        ["can", "i", "pay", "by", "visa"],
        ["can", "i", "pay", "by", "visa", "please"],
        ["can", "i", "pay", "with", "visa"],
        ["can", "i", "pay", "with", "visa", "please"],
        ["could", "i", "pay", "by", "visa"],
        ["could", "i", "pay", "by", "visa", "please"],
        ["could", "i", "pay", "with", "visa"],
        ["could", "i", "pay", "with", "visa", "please"],
        ["do", "you", "accept", "visa"],
        ["do", "you", "accept", "visa", "please"],
        ["i", "would", "like", "to", "pay", "by", "visa"],
        ["i", "would", "like", "to", "pay", "by", "visa", "please"],
        ["i", "would", "like", "to", "pay", "with", "visa"],
        ["i", "would", "like", "to", "pay", "with", "visa", "please"],
        ["i'd", "like", "to", "pay", "by", "visa"],
        ["i'd", "like", "to", "pay", "by", "visa", "please"],
        ["i'd", "like", "to", "pay", "with", "visa"],
        ["i'd", "like", "to", "pay", "with", "visa", "please"],
        ["is", "it", "possible", "to", "pay", "by", "visa"],
        ["is", "it", "possible", "to", "pay", "by", "visa", "please"],
        ["is", "it", "possible", "to", "pay", "with", "visa"],
        ["is", "it", "possible", "to", "pay", "with", "visa", "please"]]
    sent1 = ["i", "like", "to", "pay", "by", "visa"]

    ED = generic_helper.SentenceDistance()
    print ED.min_distance(sent1, sents)


def test_gensim_helper():
    """
    Test gensim helper module
    """
    documents = [["human", "machine", "interface", "for", "lab", "abc", "computer", "applications"],
                 ["a", "survey", "of", "user", "opinion", "of", "computer", "system", "response", "time"],
                 ["the", "eps", "user", "interface", "management", "system"],
                 ["system", "and", "human", "system", "engineering", "testing", "of", "eps"],
                 ["relation", "of", "user", "perceived", "response", "time", "to", "error", "measurement"],
                 ["the", "generation", "of", "random", "binary", "unordered", "trees"],
                 ["the", "intersection", "graph", "of", "paths", "in", "trees"],
                 ["graph", "minors", "iv", "widths", "of", "trees", "and", "well", "quasi", "ordering"],
                 ["graph", "minors", "a", "survey"]]
    lda_model = gensim_helper.TopicModels()
    lda_model.train_model(documents, rare_threshold=1, topic_count=10)
    print lda_model.infer_topic_distribution(["graph", "minors", "a", "survey"])


def test_ml():
    root_folder = "d:/spokenCALL/2017_v3_ASR"
    data_file = 'textProcessing_trainingKaldi_features.csv'
    feature_file = os.path.join(root_folder, 'data', data_file)
    ml.xval_loop_features(feature_file)


def main():
    test_ml()


if __name__ == '__main__':
    main()
