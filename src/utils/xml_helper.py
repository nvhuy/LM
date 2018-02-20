'''
Created on Dec 27, 2017

@author: HuyNguyen
'''
import logging
module_logger = logging.getLogger(__name__)

import os
import xml.etree.ElementTree as Et
from xml.etree import ElementTree
from xml.dom import minidom

from src.utils import fs_helper


class xmlHelper():

    def __init__(self):
        """
        :ivar __xml__: XML tree
        :ivar __plaintext__: plain text in XML format
        """
        self.__plaintext__ = None
        self.__xml__ = None

    def parse_xml_file(self, xml_file):
        """
        Parse XML file and keep XML tree in memory
        :param xml_file: XML file as input
        """
        module_logger.info('------ Parsing XML file ::: {}'.format(xml_file))
        try:
            xmlt = Et.parse(xml_file)
            self.__xml__ = xmlt.getroot()
        except:
            module_logger.exception('****** File was not parsed into XML tree ::: {}'.format(xml_file))
            self.__xml__ = None

    def print_xml(self):
        """
        Print XML content
        """
        try:
            rough_string = ElementTree.tostring(self.__xml__, 'utf-8')
            re_parsed = minidom.parseString(rough_string.replace('\n', '').replace('\r', '').replace('  ', ' '))
            print '------ XML data'
            print re_parsed.toprettyxml(indent='  ')
            print '------ EOF'
        except:
            module_logger.exception('****** No XML data to print')

    def get_value_at_path(self, key_path):
        """
        Get content of XML element
        :param key_path: XML path to element
        :return: element content
        """
        if self.__xml__ is None:
            return None
        try:
            xml_node = parse_xml_data(self.__xml__, key_path)
            if xml_node is not None:
                return xml_node.text
        except:
            module_logger.exception('****** Key path not parsed ::: {}'.format(key_path))
            return None


def summary_xml_node(xml_node):
    """
    Print a summary of XML node
    :param xml_node: a node in XML tree
    """
    try:
        print '------ summary of', xml_node.tag
        print 'Text:', xml_node.text
        print 'Attributes:', xml_node.attrib
        print 'Children:', [child.tag for child in xml_node]
    except:
        module_logger.exception('****** Invalid XML node ::: {}'.format(xml_node))


def validate_key_component(xml_node, key_component):
    """
    Validate if XML node has tag and attribute match values provided
    :param xml_node: XML node
    :param key_component: tag and attribute provided
    :return: 0 if valid
    """
    res = 0

    combined_tag = key_component.split(':::')
    if xml_node.tag != combined_tag[0]:
        return 1

    if len(combined_tag) > 1:
        for attr in combined_tag[1:]:
            attr_name_value = attr.split('===')
            if len(attr_name_value) != 2:
                return 2
            elif attr_name_value[0] not in xml_node.attrib:
                return 3
            elif attr_name_value[1] != str(xml_node.attrib[attr_name_value[0]]):
                return 4

    return res


def parse_xml_data(root_node, key_path):
    """
    key_path allows one to specify node in XML tree
    Nodes are determined first by TAG. Example XML tree
    ---------------------------------------------------
    <data>
        <country name="Liechtenstein">
            <rank>1</rank>
            <year>2008</year>
            <gdppc>141100</gdppc>
            <neighbor name="Austria" direction="E"/>
            <neighbor name="Switzerland" direction="W"/>
        </country>
        <country name="Singapore">
            <rank>4</rank>
            <year>2011</year>
            <gdppc>59900</gdppc>
            <neighbor name="Malaysia" direction="N"/>
        </country>
        <country name="Panama">
            <rank>68</rank>
            <year>2011</year>
            <gdppc>13600</gdppc>
            <neighbor name="Costa Rica" direction="W"/>
            <neighbor name="Colombia" direction="E"/>
        </country>
    </data>
    ---------------------------------------------------
    To retrieve rank of Singapore, set key_path="data/country:::name===Singapore/rank"
    To retrieve neighbor "Costa Rica" of Panama,
        set key_path="data/country:::name===Panama/neighbor:::name===Costa Rica"
    In key_path, "/" to separate node, ":::" to separate attribute, and "===" to separate attribute name from attribute
    value
    """
    module_logger.info('------ Parsing XML tree to retrieve node ::: {}'.format(key_path))

    key_path = os.path.normpath(key_path)
    key_elements = fs_helper.local_split_path(key_path)
    if not key_elements:
        tag_list = [os.path.basename(key_path)]
    else:
        tag_list = [key_elements[0][0]]
        for ke in key_elements:
            tag_list.append(ke[1])

    root = root_node

    res = validate_key_component(root, tag_list[0])
    if res != 0:
        return None

    if len(tag_list) > 1:
        for i in range(1, len(tag_list)):
            if root is not None:
                for child in root:
                    res = validate_key_component(child, tag_list[i])
                    if res == 0:
                        '''
                        ***retrieve only first child***
                        '''
                        root = child
                        break
                if res != 0:
                    break
            else:
                module_logger.error('****** Null XML node')
                break

    if res != 0:
        root = None
    return root
