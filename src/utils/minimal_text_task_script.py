
# Baseline Python3 processing script for text version of CALL shared task
#
# The script reads the XML prompt/response grammar supplied with the download
# and uses it to process the text task training data spreadsheet.
# 
# A prompt/recognition_result pair is 
#   accepted if the recognition_result is in the list of responses associated with the prompt in the grammar
#   rejected otherwise

# ---------------------------------------------------

# Example of running the script (all files in same directory as script):

# $ python3 minimal_text_task_script.py
# Read XML grammar: referenceGrammar.xml
# Reading and processing spreadsheet: textProcessing_training.csv
# 
# INCORRECT UTTERANCES (1353)
# CorrectReject    891
# GrossFalseAccept 99*3 = 297
# PlainFalseAccept 363
# RejectionRate    0.57
# 
# CORRECT UTTERANCES (3911)
# CorrectAccept    2645
# FalseReject      1266
# RejectionRate    0.32
# 
# D                1.77
# 
# Written spreadsheet: results.csv

# ---------------------------------------------------

# Weighting factor for gross false accepts

k = 3

# Files used:

# spreadsheet_in is the input spreadsheet with the prompts, recognition results, transcriptions and judgements
spreadsheet_in = 'textProcessing_training.csv'
#spreadsheet_in = 'kaldi_spreadsheet.csv'

# grammar is the XML prompt/response grammar
grammar = 'referenceGrammar.xml'

# spreadsheet_out is the output spreadsheet with the accept/reject results
spreadsheet_out = 'results.csv'

# ---------------------------------------------------

import csv
import xml.etree.ElementTree as ET

# ---------------------------------------------------

def do_all_processing():
    print("Read XML grammar: " + grammar)
    ( grammar_dic, known_prompts ) = read_grammar()
    print("Reading and processing spreadsheet: " + spreadsheet_in)
    read_and_process_spreadsheet(grammar_dic, known_prompts)
    print("\nWritten spreadsheet: " + spreadsheet_out)

# ---------------------------------------------------

# Start off by reading the XML grammar into the dictionary grammar_dic
# which associates prompts with lists of responses. 
#
# Return the dictionary and the list of prompts.

def read_grammar():
    tree = ET.parse(grammar)
    root = tree.getroot()
    dictionary = { get_prompt(unit): get_responses(unit) for unit in root.findall('prompt_unit') }
    return ( dictionary, dictionary.keys() )

def get_prompt(unit):
    prompt = unit.find('prompt').text
    return prompt

def get_responses(unit):
    return [ response.text for response in unit.findall('response') ]

# ---------------------------------------------------

# Open the input and output files and process each row, skipping the header
def read_and_process_spreadsheet(grammar_dic, known_prompts):
    out_fieldnames = ['Id', 'Prompt', 'Transcription', 'RecResult', 'Language', 'Meaning', 'Accept', 'Result']
    scores = init_scores()
    with open(spreadsheet_in, 'r', encoding="utf-8") as csv_infile:
        reader = csv.reader(csv_infile, delimiter='\t', quotechar='"')
        with open(spreadsheet_out, 'w') as csv_outfile:
            writer = csv.DictWriter(csv_outfile, fieldnames=out_fieldnames, delimiter='\t', quotechar='"')
            writer.writeheader()
            for row in reader:
                # Skip header row
                if ( not is_header_row(row) ):
                    process_spreadsheet_row(row, grammar_dic, known_prompts, writer, scores)
    print_scores(scores)

def init_scores():
    return {'CorrectAccept': 0, 'GrossFalseAccept': 0, 'PlainFalseAccept': 0, 'CorrectReject': 0, 'FalseReject': 0}

def is_header_row(row):
    return ( row[0] == 'Id' )

def process_spreadsheet_row(row, grammar_dic, known_prompts, writer, scores):
    id = row[0]
    prompt = row[1]
    rec_result = row[2]
    transcription = row[3]
    language_correct_gold_standard = row[4]
    meaning_correct_gold_standard = row[5]
    decision = classification_according_to_xml_grammar(prompt, rec_result, grammar_dic, known_prompts)
    result = score_decision(decision, language_correct_gold_standard, meaning_correct_gold_standard, scores)

    writer.writerow({'Id': id, 
                     'Prompt': prompt, 
                     'RecResult': rec_result,
                     'Transcription': transcription,
                     'Language': language_correct_gold_standard, 
                     'Meaning': meaning_correct_gold_standard, 
                     'Accept': decision, 
                     'Result': result})

# Look up <prompt, response> pair in dictionary.
# Return 'accept' if it's there, 'reject' otherwise.
# Print an error if the prompt isn't mentioned in the dictionary.
def classification_according_to_xml_grammar(prompt, response, grammar_dic, known_prompts):
    if ( prompt in known_prompts ):
        valid_responses = grammar_dic[prompt]
        if ( response in valid_responses ):
            return 'accept'
        else:
            return 'reject'
    else:
        print("*** Error: prompt not in XML dictionary: '" + prompt + "'") 
        return False

# Compare decision with gold standard judgements for language and meaning
def score_decision(decision, language_correct_gs, meaning_correct_gs, scores):
    if ( decision == 'accept' and language_correct_gs == 'correct' ):
        result = 'CorrectAccept'
    elif ( decision == 'accept' and meaning_correct_gs == 'incorrect' ):
        result = 'GrossFalseAccept'
    elif ( decision == 'accept' ):
        result = 'PlainFalseAccept'
    elif ( decision == 'reject' and language_correct_gs == 'incorrect' ):
        result = 'CorrectReject'
    else:
        result = 'FalseReject'
    scores[result] = scores[result] + 1
    return result

def print_scores(scores):
    CA = scores['CorrectAccept']
    GFA = scores['GrossFalseAccept']
    PFA = scores['PlainFalseAccept']
    CR = scores['CorrectReject']
    FR = scores['FalseReject']

    FA = PFA + k * GFA
    Correct = CA + FR
    Incorrect = CR + GFA + PFA
    
    if ( CR + FA ) > 0 :
        IncorrectRejectionRate = CR / ( CR + FA )
    else:
        IncorrectRejectionRate = 'undefined'

    if ( FR + CA ) > 0 :
        CorrectRejectionRate = FR / ( FR + CA )
    else:
        CorrectRejectionRate = 'undefined'

    if ( CorrectRejectionRate != 'undefined' and IncorrectRejectionRate != 'undefined' ) :
        D = IncorrectRejectionRate / CorrectRejectionRate 
    else:
        D = 'undefined'

    print('\nINCORRECT UTTERANCES (' + str(Incorrect) + ')' )
    print('CorrectReject    ' + str(CR) )
    print('GrossFalseAccept ' + str(GFA) + '*' + str(k) + ' = ' + str(GFA * k) )
    print('PlainFalseAccept ' + str(PFA) )
    print('RejectionRate    ' + two_digits(IncorrectRejectionRate) )

    print('\nCORRECT UTTERANCES (' + str(Correct) + ')')
    print('CorrectAccept    ' + str(CA) )
    print('FalseReject      ' + str(FR) )
    print('RejectionRate    ' + two_digits(CorrectRejectionRate) )

    print('\nD                ' + two_digits(D) )

def two_digits(x):
    if x == 'undefined':
        return 'undefined'
    else:
        return ( "%.2f" % x )

# ---------------------------------------------------

do_all_processing()
