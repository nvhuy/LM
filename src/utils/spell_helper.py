"""
Created on Jan 2, 2018

@author: hnguyen6
"""
import logging
module_logger = logging.getLogger(__name__)

import enchant
from enchant.checker import SpellChecker


class EnchantSpell:

    def __init__(self):
        print enchant.Broker().list_dicts()
        print enchant.Broker().list_languages()
        self.__spell_checker__ = SpellChecker(lang='en_US')

    def check_token(self, token):
        """
        Check spelling of a single word
        :param token: input word
        :return: True/False
        """
        return self.__spell_checker__.check(token) or self.__spell_checker__.check(token.capitalize())

    def check_tokens(self, tokens):
        """
        Check spelling of a list of words
        :param tokens: input word list
        :return: a list of index of mis-spelled words
        """
        error_idx = []
        for tid, token in enumerate(tokens):
            if not self.check_token(token):
                error_idx.append(tid)
        return error_idx

    def suggest_correction(self, token):
        """
        Suggest correction to a mis-spelled word
        :param token: input word
        :return: list of suggestion
        """
        return self.__spell_checker__.suggest(token)
