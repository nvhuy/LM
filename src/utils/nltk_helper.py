"""
Created on Dec 31, 2017

@author: HuyNguyen
"""
import logging
module_logger = logging.getLogger(__name__)

import nltk


def segment_text(raw_text):
    """
    Segment input text into sentences
    """
    sentences = nltk.sent_tokenize(raw_text)
    return sentences


def tokenize_sentence(sent):
    """
    Tokenize text into words
    :param sent: input sentence
    :return: word list
    """
    tokens = nltk.word_tokenize(sent, preserve_line=True)
    return tokens


def tag_sentence(sent):
    """
    Tokenize text into words and tag words
    :param sent: input sentence
    :return: (word, tag) list
    """
    tokens = nltk.word_tokenize(sent, preserve_line=True)
    token_pos = nltk.pos_tag(tokens)
    return [pos[1] for pos in token_pos]


def tag_tokens(tokens):
    """
    Tag tokens
    :param tokens: token list
    :return: tag list
    """
    token_pos = nltk.pos_tag(tokens)
    return [pos[1] for pos in token_pos]


def common_ngrams(tokens1, tokens2, n_order, padding=False):
    """
    Extract common ngrams between two token lists
    :param tokens1: first token list
    :param tokens2: second token list
    :param n_order: ngram order
    :param padding: start and end of sentence
    """
    pad_left_sym = '<sos>'
    pad_right_sym = '<eos>'
    ngrams1 = nltk.ngrams(sequence=tokens1, n=n_order, pad_left=padding, pad_right=padding,
                          left_pad_symbol=pad_left_sym, right_pad_symbol=pad_right_sym)
    ngrams2 = nltk.ngrams(sequence=tokens2, n=n_order, pad_left=padding, pad_right=padding,
                          left_pad_symbol=pad_left_sym, right_pad_symbol=pad_right_sym)
    commons = set(ngrams1) & set(ngrams2)
    return commons


def extract_production_rules(parse_string, include_leaf=False):
    """
    Extract production rules from a parse string
    :param parse_string: input parse string
    :param include_leaf: if True then include leaf productions
    :return: list of production rules
    """
    t = nltk.tree.Tree.fromstring(parse_string)
    prods = t.productions()
    prod_sign_list = []
    for p in prods:
        if include_leaf or p.is_nonlexical():
            p_sign = p.unicode_repr().replace(' -> ', '>').replace(' ', '|')
            prod_sign_list.append(p_sign)

    return prod_sign_list
