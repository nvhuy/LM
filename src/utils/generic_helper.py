"""
Created on Jan 4, 2018

@author: HuyNguyen
"""
import codecs
import collections
import re
import logging
module_logger = logging.getLogger(__name__)


import edit_distance


class SentenceDistance:
    """
    Wrapper object to edit_distance
    """
    def __init__(self):
        self.__ed__ = edit_distance.SequenceMatcher(action_function=edit_distance.highest_match_action)

    def distance(self, sent1, sent2):
        """
        Edit distance between two sentences
        :param sent1: list of tokens (string)
        :param sent2: list of tokens (string)
        :return: edit distance
        """
        self.__ed__.set_seq1(sent1)
        self.__ed__.set_seq2(sent2)
        return self.__ed__.distance()

    def edit_ops(self, sent1, sent2):
        """
        Edit distance between two sentences
        :param sent1: list of tokens (string)
        :param sent2: list of tokens (string)
        :return: edit distance
        """
        self.__ed__.set_seq1(sent1)
        self.__ed__.set_seq2(sent2)
        self.__ed__.distance()
        ops = self.__ed__.get_opcodes()
        return ops

    def min_distance(self, sent1, sents):
        """
        Min edit distance between input to sentence to a set of references
        :param sent1: list of tokens (string)
        :param sents: list of sentences
        :return: edit distance
        """
        self.__ed__.set_seq1(sent1)
        min_score = (9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0)
        min_ref = None
        for sent in sents:
            char_len = sum([len(tok) for tok in sent])
            self.__ed__.set_seq2(sent)
            dist = self.__ed__.distance()
            ops = self.__ed__.get_opcodes()
            op_dict = collections.defaultdict(int)
            for op in ops:
                op_dict[op[0]] += 1
            score = (dist, -op_dict['equal'], len(sent), op_dict['insert'], op_dict['delete'], op_dict['replace'],
                     char_len)
            if min_score > score:
                min_score = score
                min_ref = sent
        return min_score[0], min_ref


def distance_sentences(sent1, sent2):
    """
    Edit distance between two sentences
    :param sent1: list of tokens (string)
    :param sent2: list of tokens (string)
    :return: edit distance
    """
    s = edit_distance.SequenceMatcher(a=sent1, b=sent2, action_function=edit_distance.highest_match_action)
    return s.distance()


def read_parsed_line(parsed_line):
    """
    Extract parsing output from parsed line
    """
    parsed = {}
    tup = parsed_line.strip().split('\t')
    parsed['tokens'] = tup[5].split()
    parsed['pos'] = tup[6].split()
    parsed['lemma'] = tup[7].split()
    parsed['parse'] = tup[10]
    parsed['dependency'] = tup[11].split('::')
    parsed['score'] = float(tup[12])
    return parsed


def read_responses_from_parse(parse_file, id_file):
    """
    Read parse output of responses
    :param parse_file: parse file
    :param id_file: id file
    """
    module_logger.info('------ Read parse output ::: {} ::: {}'.format(parse_file, id_file))

    try:
        rf = codecs.open(id_file, mode='rb', encoding='utf-8')
        responses_id = rf.readlines()
        rf.close()

        rf = codecs.open(parse_file, mode='rb', encoding='utf-8')
        responses_parsed = rf.readlines()
        rf.close()

        if len(responses_id) != len(responses_parsed):
            module_logger.error('****** Not matched ID file and parse file ::: {} ::: {}'
                                .format(len(responses_id), len(responses_parsed)))
            return None

        response_parsed_dict = {}
        for idx, parsed_id in enumerate(responses_id):
            parsed_line = responses_parsed[idx].strip()
            response_parsed_dict[parsed_id.strip()] = read_parsed_line(parsed_line)
    except:
        module_logger.exception('****** Failed reading parse output')
        response_parsed_dict = None

    return response_parsed_dict


def reduce_dependency_rules(dependency_rules):
    """
    Remove token index from dependency triples
    :param dependency_rules: list of dependency triples
    """
    reduced_rules = []

    patt1 = re.compile('-[0-9]+, ')
    patt2 = re.compile('-[0-9]+\)')
    for dep in dependency_rules:
        red = re.sub(patt1, ',', dep)
        red = re.sub(patt2, ')', red)
        reduced_rules.append(red)
    return reduced_rules


def create_head_word_patterns():
    """
    Prepare a list of head word patterns to re-use
    :return: pattern list
    """
    head_pattern_list = [re.compile('^thanks\sa\slot\s'), re.compile('^thank\syou\s'), re.compile('^thanks\s'),
                         re.compile('^thank\s'), re.compile('^no\sthank\syou\s'), re.compile('^no\sthank\s'),
                         re.compile('^yes\s'), re.compile('^no\s'), re.compile('^i\sthink\s'),
                         re.compile('^excuse\sme\s'), re.compile('^sorry\s')]
    return head_pattern_list


def remove_head_words_response(response, head_pattern_list):
    """
    Remove head words, e.g., thank, yes, no... Only when response has more than 5 words
    :param response: input string
    :param head_pattern_list: list of head word patterns
    :return: new string
    """
    n = len(response.split())
    if n <= 2:
        return response
    elif not response.startswith('no') and not response.startswith('yes') and n <= 5:
        return response
    else:
        for patt in head_pattern_list:
            response = re.sub(patt, '', response)
    return response


def clean_transcript(response, filler_pattern_list, head_pattern_list):
    """
    Clean transcription
    :param response: input string
    :param filler_pattern_list: list of filler patterns
    :param head_pattern_list: list of head word patterns
    :return: new string
    """
    cleaned = remove_unrecognized_tokens(response)
    cleaned = remove_word_repetition(cleaned)
    cleaned = remove_filler_sounds(cleaned, filler_pattern_list)
    cleaned = remove_head_words_response(cleaned, head_pattern_list)
    return cleaned


def remove_unrecognized_tokens(response):
    """
    Remove tokens marked as unrecognized
    :param response: input string
    :return: new string
    """
    res_cleaned = response.replace('***', '').replace('-***', '').replace('*silence', '')\
        .replace('*v', '').replace('*z', '').replace('*a', '').replace('*x', '').replace('ggg', '').strip()
    res_cleaned = re.sub('\s+', ' ', res_cleaned)
    return res_cleaned


def create_filler_sound_patterns():
    """
    Prepare a list of filler sound patterns to re-use
    :return: pattern list
    """
    filler_pattern_list = [re.compile('e+[hm]+$'), re.compile('u+[hm]+$'), re.compile('o+h+$'), re.compile('a+h+$'),
                           re.compile('mm+$'), re.compile('e+u+h+[hm]*$'), re.compile('h+m+$')]
    return filler_pattern_list


def remove_filler_sounds(response, filler_pattern_list):
    """
    Remove filler sounds from text
    :param response: input text
    :param filler_pattern_list: list of filler patterns
    :return: new string
    """
    tokens = response.split()
    clean_tokens = []
    for tok in tokens:
        is_filler = False
        for filler_pattern in filler_pattern_list:
            if re.match(filler_pattern, tok) is not None:
                is_filler = True
                break
        if not is_filler:
            clean_tokens.append(tok)

    return ' '.join(clean_tokens)


def remove_word_repetition(response):
    """
    Remove repeated words in response
    :param response: input string
    :return: response with no word repetition
    """
    tokens = response.split()
    unique_tokens = {}
    clean_response = []
    for idx in range(len(tokens)-1, -1, -1):
        tok = tokens[idx]
        if tok not in unique_tokens:
            unique_tokens[tok] = idx
            clean_response.append(tok)
    clean_response.reverse()
    return ' '.join(clean_response)


def read_verb_long_short_list(ls_file):
    """
    Read list of long short form mapping of verb
    :param ls_file: verb list file
    :return: list of verb long short form
    """
    verb_list = []
    vfile = open(ls_file)
    for ln in vfile.readlines():
        if ln.strip() != '':
            verb_list.append(ln.strip().split(' ::: '))
    return verb_list


def replace_contracted_verb(response, verb_list):
    """
    Replace long-short form of verbs
    :param response: input text
    :param verb_list: list of verb long short form
    :return: list of responses
    """
    response_sos = '<sos>' + response
    new_responses = set()
    new_responses.add(response)
    for ls in verb_list:
        long_form = ls[0]
        long_first = long_form.split()[0]
        short_form = ls[1]
        if long_first in ['i', 'we', 'you', 'he', 'she', 'they', 'it']:
            long_form_sos = '<sos>' + long_form
            new_responses.add(response_sos.replace(long_form_sos, short_form).replace('<sos>', ''))
        else:
            new_responses.add(response.replace(long_form, short_form))
        new_responses.add(response.replace(short_form, long_form))

    return new_responses


def init_scores():
    return {'CorrectAccept': 0, 'GrossFalseAccept': 0, 'PlainFalseAccept': 0, 'CorrectReject': 0, 'FalseReject': 0}


def score_decision(decision, language_correct_gs, meaning_correct_gs, scores):
    """
    Compare decision with gold standard judgements for language and meaning
    """
    if decision == 'accept' and language_correct_gs == 'correct':
        result = 'CorrectAccept'
    elif decision == 'accept' and meaning_correct_gs == 'incorrect':
        result = 'GrossFalseAccept'
    elif decision == 'accept':
        result = 'PlainFalseAccept'
    elif decision == 'reject' and language_correct_gs == 'incorrect':
        result = 'CorrectReject'
    else:
        result = 'FalseReject'
    scores[result] = scores[result] + 1.0
    return result


def print_scores(scores, k):
    """
    :param scores: scores as dict {score name: value}
    :param k: Weighting factor for gross false accepts
    """
    print scores
    CA = scores['CorrectAccept']
    GFA = scores['GrossFalseAccept']
    PFA = scores['PlainFalseAccept']
    CR = scores['CorrectReject']
    FR = scores['FalseReject']

    FA = PFA + k * GFA
    Correct = CA + FR
    Incorrect = CR + GFA + PFA

    if (CR + FA) > 0:
        IncorrectRejectionRate = CR / (CR + FA)
    else:
        IncorrectRejectionRate = -1.0

    if (FR + CA) > 0:
        CorrectRejectionRate = FR / (FR + CA)
    else:
        CorrectRejectionRate = -1.0

    print IncorrectRejectionRate, CorrectRejectionRate
    if CorrectRejectionRate >= 0 and IncorrectRejectionRate >= 0:
        D = IncorrectRejectionRate / CorrectRejectionRate
    else:
        D = 0

    print('\nINCORRECT UTTERANCES (' + str(Incorrect) + ')')
    print('CorrectReject    ' + str(CR))
    print('GrossFalseAccept ' + str(GFA) + '*' + str(k) + ' = ' + str(GFA * k))
    print('PlainFalseAccept ' + str(PFA))
    print('RejectionRate    ' + two_digits(IncorrectRejectionRate))

    print('\nCORRECT UTTERANCES (' + str(Correct) + ')')
    print('CorrectAccept    ' + str(CA))
    print('FalseReject      ' + str(FR))
    print('RejectionRate    ' + two_digits(CorrectRejectionRate))

    print('\nD                ' + two_digits(D))
    return D


def two_digits(x):
    if x == 'undefined':
        return 'undefined'
    else:
        return "%.2f" % x


def expand_string(input_list):
    """
    c ( a | b ) => ca, cb
    :param input_list: input list of strings
    :return: multiple lists of strings
    """
    open_count_list = []
    open_before = 0
    or1_count = 0
    open_first_closed = False
    for tok in input_list:
        if not open_first_closed:
            if tok == '(':
                open_before += 1
            elif tok == '|':
                if open_before == 1:
                    or1_count += 1
            elif tok == ')':
                open_before -= 1
                if open_before == 0:
                    open_first_closed = True
        else:
            open_before = -1
        open_count_list.append(open_before)

    # print input_list
    # print open_count_list

    new_lists = []
    for i in range(or1_count + 1):
        new_lists.append([])

    new_id = 0
    for tid, tok in enumerate(input_list):
        if open_count_list[tid] == -1:
            for l in new_lists:
                l.append(tok)
        elif open_count_list[tid] == 0:
            if tok != ')':
                for l in new_lists:
                    l.append(tok)
            else:
                new_id = 0
        else:
            if not (tok == '(' and open_count_list[tid] == 1):
                if open_count_list[tid] == 1 and tok == '|':
                    new_id += 1
                else:
                    new_lists[new_id].append(tok)

    final_list = []
    for nl in new_lists:
        if '(' in nl:
            more_list = expand_string(nl)
            for ml in more_list:
                final_list.append(ml)
        else:
            final_list.append(nl)

    # for fl in final_list:
    #     print fl
    # print '*****'

    return final_list


def calculate_wer(ref, hyp, debug=False):
    """
    :param ref: reference sentence
    :param hyp: hypothesis sentence
    :param debug: True or False
    :return: WER
    """
    r = ref.split()
    h = hyp.split()
    # costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for _ in range(len(h) + 1)] for _ in range(len(r) + 1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for _ in range(len(h) + 1)] for _ in range(len(r) + 1)]

    DEL_PENALTY = 1
    INS_PENALTY = 1
    SUB_PENALTY = 1

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r) + 1):
        costs[i][0] = DEL_PENALTY * i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY  # penalty is always 1
                insertionCost = costs[i][j - 1] + INS_PENALTY  # penalty is always 1
                deletionCost = costs[i - 1][j] + DEL_PENALTY  # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0.0
    numDel = 0.0
    numIns = 0.0
    numCor = 0.0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    if i == 0:
        numIns = j
        if j == 0:
            wer_result = 0.0
        else:
            wer_result = 1.0
    else:
        while i > 0 or j > 0:
            if backtrace[i][j] == OP_OK:
                numCor += 1
                i -= 1
                j -= 1
                if debug:
                    lines.append("OK\t" + r[i] + "\t" + h[j])
            elif backtrace[i][j] == OP_SUB:
                numSub += 1
                i -= 1
                j -= 1
                if debug:
                    lines.append("SUB\t" + r[i] + "\t" + h[j])
            elif backtrace[i][j] == OP_INS:
                numIns += 1
                j -= 1
                if debug:
                    lines.append("INS\t" + "****" + "\t" + h[j])
            elif backtrace[i][j] == OP_DEL:
                numDel += 1
                i -= 1
                if debug:
                    lines.append("DEL\t" + r[i] + "\t" + "****")

        wer_result = round((numSub + numDel + numIns) / len(r), 3)
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
    # return (numSub + numDel + numIns) / (float)(len(r))
    return {'WER': wer_result, 'Cor': numCor, 'Sub': numSub, 'Ins': numIns, 'Del': numDel}
