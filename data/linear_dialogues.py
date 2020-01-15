#!/usr/bin/python3

"""
@author Johannes Schäfer - IwiSt, University of Hildesheim
"""

from data import read
import string
from xml.sax.saxutils import unescape
import json
import sys


class LinearDialogue(object):

    def __init__(self, top_comment, offensive_comment, preceding_context, following_context):
        self.top_comment = top_comment
        self.offensive_comment = offensive_comment
        self.preceding_context = preceding_context
        self.following_context = following_context

    def __len__(self):
        return 1 + len(self.preceding_context) + len(self.following_context)

    def has_empty_comments(self):
        if self.offensive_comment[0].text is None or not self.offensive_comment[0].text.strip():
            return True
        for context in (self.preceding_context, self.following_context):
            for comment in context:
                if (comment[0].text is None) or \
                   (not comment[0].text.strip()) or \
                   (comment[0].text.strip() in ('[deleted]', '[removed]')):
                    return True
        return False

    def iterate_comments(self):
        for comment in self.preceding_context:
            yield comment
        yield self.offensive_comment
        for comment in self.following_context:
            yield comment


class LanguageFilter(object):

    def __init__(self, words_filename, threshold=0.3):
        self.words = {w: True for w in open(words_filename, encoding='utf-8').read().split('\n')[:-1]}
        self.threshold = threshold

    def check_ld(self, linear_dialogue):
        for comment in linear_dialogue.iterate_comments():
            words = ''.join([c for c in comment[0].text.strip() if c not in string.punctuation]).split(' ')
            if len([True for word in words if word in self.words]) / float(len(words)) < self.threshold:
                return False
        return True


def find_offensive_comments_with_ancestors(node, depth, ancestors=None):
    ancestors = ancestors or []
    try:
        if node.attrib['off'].strip('[] ') == 'True':
            yield (node, depth), ancestors
    except KeyError:
        pass
    new_ancestors = ancestors + [(node, depth)]
    for child in node:
        for res in find_offensive_comments_with_ancestors(child, depth + 1, ancestors=new_ancestors):
            yield res


def find_descendants(node, depth, target_num_descendants, descendants):
    new_descendants = descendants + [(node, depth)]
    if len(new_descendants) == target_num_descendants:
        yield new_descendants
    else:
        for child in node:
            for res in find_descendants(child, depth + 1, target_num_descendants, new_descendants):
                yield res


def get_linear_dialogues(xml_corpus, window_size_pre=3, window_size_post=3):
    for submission in xml_corpus:
        for comment in submission:
            for offensive_comment, ancestors in find_offensive_comments_with_ancestors(comment, 1):
                if len(ancestors) < window_size_pre:
                    # preceding context too short
                    continue
                top = submission
                off, off_depth = offensive_comment

                for child in off:
                    for descendants in find_descendants(child, off_depth + 1, window_size_post, []):
                        # note: following context length check is included in function
                        yield LinearDialogue(top, offensive_comment, ancestors[-window_size_pre:], descendants)


def get_main_post_text(submission):
    for post in submission:
        if post.tag == 'main_post':
            for comment in post:
                try:
                    # case: main_post contains a link
                    return '0', comment.attrib['url'], comment.attrib.get('author', '[unknown]'),\
                           comment.attrib.get('date', '[unknown]')
                except KeyError:
                    # case: main_post contains a comment
                    return comment.attrib.get('off_p', '0'), comment.text, comment.attrib.get('author', '[unknown]'),\
                           comment.attrib.get('date', '[unknown]')
    return '0', None, '[unknown]', '[unknown]'


def write_reddit_linear_dialogues_json(outfile_path, linear_dialogues):
    data = {}
    for linear_dialogue in linear_dialogues:
        turns = []
        conv_keys = []

        # save main post
        top = linear_dialogue.top_comment
        main_post_off, main_post, main_post_author, main_post_date = get_main_post_text(top)
        main_post = main_post or '-'
        main_post = unescape(main_post.strip()) or '-'
        main_post_off = main_post_off.strip('[] ') or '0'
        main_post_date = main_post_date or '-'
        turns.append(('0', top.attrib['id'], main_post_author, main_post_off, main_post_date, main_post))
        conv_keys.append(top.attrib['id'])

        # write comments in window
        for commend_depth_pair in linear_dialogue.iterate_comments():
            comment, comment_depth = commend_depth_pair
            comment_off = comment.attrib.get('off_p', '0')
            comment_text = comment.text or '-'
            comment_text = unescape(comment_text.strip())
            turns.append((str(comment_depth), comment.attrib['id'], comment.attrib.get('author', '[unknown]'),
                          comment_off.strip('[] '), comment.attrib.get('date', '[unknown]'), comment_text))
            conv_keys.append(comment.attrib['id'])
        key = ' '.join(conv_keys)
        data[key] = turns

    with open(outfile_path, mode='w', encoding='utf-8') as outfile:
        json.dump(data, outfile)


def extract_linear_dialogues_from_corpus(xml_corpus_path, output_ld_path, language_filter_path=False, window_size=3):
    xml_corpus = read.read_reddit_xml_corpus(xml_corpus_path)

    if language_filter_path:
        language_filter = LanguageFilter(language_filter_path).check_ld
    else:
        def language_filter(_): return True

    linear_dialogues = []
    for ld in get_linear_dialogues(xml_corpus, window_size_pre=window_size, window_size_post=window_size):
        if not ld.has_empty_comments():
            if language_filter(ld):
                linear_dialogues.append(ld)
    write_reddit_linear_dialogues_json(output_ld_path, linear_dialogues)
