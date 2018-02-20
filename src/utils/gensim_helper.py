"""
Created on Jan 2, 2018

@author: HuyNguyen
"""
import logging
import numpy
from collections import defaultdict
module_logger = logging.getLogger(__name__)

import gensim


class WordVector:

    def __init__(self, dim=30, skip_gram=1, neg_samlping=0, window=4, min_count=1):
        self.__dimension__ = dim
        self.__skip_gram__ = skip_gram
        self.__negative_sampling__ = neg_samlping
        self.__window__ = window
        self.__min_count__ = min_count

        self.__w2v_model__ = None

    def set_parameters(self, dim=30, skip_gram=1, neg_samlping=0, window=4, min_count=1):
        """
        Set dimension of word vector
        :param dim: the dimensionality of the feature vectors
        :param skip_gram: defines the training algorithm. 1 -- skip-gram is employed, 0 -- CBOW is used
        :param neg_samlping: if non-zero, negative sampling will be used. Else hierarchical softmax will
        be used for model training
        :param window: maximum distance between the current and predicted word within a sentence
        :param min_count: ignore all words with total frequency lower than this
        """
        self.__dimension__ = dim
        self.__skip_gram__ = skip_gram
        self.__negative_sampling__ = neg_samlping
        self.__window__ = window
        self.__min_count__ = min_count

    def train_model(self, sentence_list, num_procs=4):
        """
        Train word2vec model using number of processes
        :param sentence_list: sentence list. Each sentence is a list of tokens
        :param num_procs: number of processes
        :return: word2vec model
        """
        module_logger.info('------ Training word2vec model')
        w2v_model = None
        try:
            hierarchical_softmax = 1
            w2v_model = gensim.models.Word2Vec(sentences=sentence_list,
                                               size=self.__dimension__,
                                               window=self.__window__,
                                               min_count=self.__min_count__,
                                               workers=num_procs,
                                               sg=self.__skip_gram__,
                                               hs=hierarchical_softmax,
                                               negative=self.__negative_sampling__,
                                               batch_words=50)
        except:
            module_logger.exception('****** Failed training word2vec model')
        return w2v_model

    def train_save_model(self, sentence_list, model_file, num_procs=4):
        """
        Train word2vec model and save to file
        :param sentence_list: sentence list. Each sentence is a list of tokens
        :param model_file: path to output file
        :param num_procs: number of processes
        """
        w2v_model = self.train_model(sentence_list, num_procs)
        if w2v_model is not None:
            module_logger.info('------ Saving word2vec model to file ::: {}'.format(model_file))
            try:
                w2v_model.save(model_file)
            except:
                module_logger.exception('****** Failed saving word2vec model')

    def load_model(self, model_file):
        """
        Load word2vec model from file
        :param model_file: path to output file
        """
        module_logger.info('------ Loading word2vec model from file ::: {}'.format(model_file))
        try:
            self.__w2v_model__ = gensim.models.Word2Vec.load(model_file)
        except:
            module_logger.exception('****** Failed loading word2vec model')
            self.__w2v_model__ = None

    def score_sentence(self, sent):
        """
        Calculate log probability of a sentence
        :param sent: input sentence
        :return: log probability
        """
        try:
            lp = self.__w2v_model__.score(sent)
        except:
            lp = 0
        return lp

    def sentence_similarity(self, sent1, sent2):
        """
        Calculate cosine similarity of two sentences
        :param sent1: first sentence
        :param sent2: second sentence
        :return: similarity score
        """
        try:
            vs = self.__w2v_model__.vector_size
            wv = self.__w2v_model__.wv

            avg1 = numpy.full(vs, 0.0000000001)
            for tok in sent1:
                if tok in wv:
                    avg1 += wv[tok]
            avg1 = avg1 / len(sent1)

            avg2 = numpy.full(vs, 0.0000000001)
            for tok in sent2:
                if tok in wv:
                    avg2 += wv[tok]
            avg2 = avg2 / len(sent1)

            lp = numpy.dot(gensim.matutils.unitvec(avg1), gensim.matutils.unitvec(avg2))
        except:
            lp = 0
        return lp


class D2C:
    """
    Convert text list to corpus
    """
    def __init__(self):
        self.__dictionary__ = None

    def build_dictionary(self, documents, rare_threshold):
        """
        Initialize document list
        :param documents: list of list of word (string)
        :param rare_threshold: remove words with occurrence less than threshold
        """
        frequency = defaultdict(int)
        for doc in documents:
            for wrd in doc:
                frequency[wrd] += 1
        texts = [[wrd for wrd in doc if frequency[wrd] >= rare_threshold] for doc in documents]
        self.__dictionary__ = gensim.corpora.Dictionary(texts)

    def save_dictionary(self, dict_file):
        """
        Save dictionary to file
        :param dict_file: output file
        """
        self.__dictionary__.save(dict_file)

    def load_dictionary(self, dict_file):
        """
        Load dictionary from file
        :param dict_file: input file
        """
        self.__dictionary__ = gensim.corpora.Dictionary.load(dict_file)

    def create_bow_vector(self, doc):
        """
        Create BOW vector of input document
        :param doc: list of word (string)
        :return: bow vector
        """
        text = [wrd for wrd in doc if wrd in self.__dictionary__.token2id]
        bow_vector = self.__dictionary__.doc2bow(text)
        return bow_vector

    def create_bow_matrix(self, documents):
        """
        Create bow matrix (corpus) from list of documents
        :param documents: list of list of word (string)
        :return: bow matrix
        """
        bow_matrix = [self.create_bow_vector(doc) for doc in documents]
        return bow_matrix


class TopicModels:
    """
    Topic modeling helper
    """
    def __init__(self):
        self.__d2c__ = D2C()
        self.__lda_model__ = None

    def train_model(self, documents, rare_threshold=3, topic_count=50):
        """
        :param documents: list of list of word (string)
        :param rare_threshold: remove words with occurrence no more than threshold
        :param topic_count: number of topics
        """
        self.__d2c__.build_dictionary(documents, rare_threshold)
        bow_matrix = self.__d2c__.create_bow_matrix(documents)
        self.__lda_model__ = gensim.models.LdaModel(corpus=bow_matrix, num_topics=topic_count,
                                                    iterations=1000, passes=200,
                                                    id2word=self.__d2c__.__dictionary__,
                                                    minimum_probability=0.005)

    def get_topic_count(self):
        """
        Get number of topics
        :return: number of topics
        """
        num_topics = 0
        if self.__lda_model__ is not None:
            num_topics = self.__lda_model__.num_topics
        return num_topics

    def save_model(self, model_file, dict_file):
        """
        Save LDA model to file
        :param model_file: model file
        :param dict_file: dictionary file
        """
        self.__d2c__.save_dictionary(dict_file)
        self.__lda_model__.save(model_file)

    def load_model(self, model_file, dict_file):
        """
        Load LDA model from file
        :param model_file: model file
        :param dict_file: dictionary file
        """
        self.__d2c__.load_dictionary(dict_file)
        self.__lda_model__ = gensim.models.LdaModel.load(model_file)

    def infer_topic_distribution(self, doc):
        """
        Get topic distribution given document input
        :param doc: input document
        :return: topic distribution
        """
        doc_bow = self.__d2c__.create_bow_vector(doc)
        td = self.__lda_model__.get_document_topics(doc_bow)
        td_float = [0.0] * self.__lda_model__.num_topics
        for top in td:
            td_float[top[0]] = numpy.asscalar(top[1])
        return td_float

    def topic_similarity(self, doc1, doc2):
        """
        Calculate topic similarity between two documents input
        :param doc1: list of words (string)
        :param doc2: list of words (string)
        :return: similarity score
        """
        td1 = self.infer_topic_distribution(doc1)
        td2 = self.infer_topic_distribution(doc2)
        vec1 = numpy.asarray(td1)
        vec2 = numpy.asarray(td2)
        sim = numpy.dot(gensim.matutils.unitvec(vec1), gensim.matutils.unitvec(vec2))
        return sim
