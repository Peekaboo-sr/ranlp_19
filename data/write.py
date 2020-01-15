#!/usr/bin/python3

"""
@author Johannes Schäfer - IwiSt, University of Hildesheim
"""

import sys
import xml.etree.ElementTree as ET


def add_prediction_to_descendants(node, predictions, index):
    if node.tag == 'comment':
        if node.text:
            node.set('off', str(predictions[index][0] >= 0.5))
            node.set('off_p', str(predictions[index][0]))
            index += 1
    for child in node:
        index = add_prediction_to_descendants(child, predictions, index)
    return index


def write_reddit_xml_corpus(outfile_path, xml_corpus_path, predictions, xml_corpus_dtd):
    sys.stderr.write('Writing output XML corpus...\n')
    index = 0
    xml_corpus = ET.parse(xml_corpus_path).getroot()
    add_prediction_to_descendants(xml_corpus, predictions, index)
    xml_corpus_text = ET.tostring(xml_corpus, encoding='unicode')
    # prettify xml
    xml_corpus_text = xml_corpus_text.replace('<', '\n<').replace('>', '>\n')
    xml_corpus_text = xml_corpus_text.replace('\n\n', '\n').replace('\n\n', '\n').strip() + '\n'
    text_s = "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>\n"
    dtd_text = open(xml_corpus_dtd, 'r', encoding='utf-8').read().strip() + '\n'
    text_e = xml_corpus_text
    outtext = text_s + dtd_text + text_e
    with open(outfile_path, 'w', encoding='utf-8') as outfile:
        outfile.write(outtext)
    sys.stderr.write('... done.\n')
