#!/usr/bin/python3

"""
@author Johannes Schäfer - IwiSt, University of Hildesheim

Contains functions to reformat the downloaded reddit posts into a unified, tree-structured XML corpus.
"""


import os
import sys
import pickle
import codecs
import unicodedata
from abc import ABCMeta, abstractmethod
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape, quoteattr

PATH_TO_SUBREDDITCORPUS_DTD = os.path.join(os.path.dirname(os.path.realpath(__file__)),  'subredditcorpus.dtd')


def remove_control_characters(s):
    """Function to remove control characters of a given string, as these are not valid for xml texts.
    :param s: given string
    :return: the string without control characters
    """
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch=='\n')


def write_xml(filename, xml_element):
    """Write a given xml element (with its children recursively) into a file with a given filename.
    """
    with codecs.open(filename, mode='w', encoding='utf-8') as outfile:
        outfile.write("<?xml version='1.0' encoding='UTF-8' standalone='yes'?>\n")
        outfile.write(codecs.open(PATH_TO_SUBREDDITCORPUS_DTD, encoding='utf-8').read())
        outfile.write(remove_control_characters(ET.tostring(xml_element, encoding='unicode'))
                      .replace('<', '\n<').replace('>', '>\n').replace('\n\n','\n').replace('\n\n','\n').strip() + '\n')


class RedditSubmission(object):
    """Class for a reddit submission: a top-level post with several replies (comments) in a tree structure.
    """

    def __init__(self, submission_id):
        self.submission_id = submission_id
        self.title = None
        self.date = None
        self.main_post = None
        self.comments = []

    def add_main_post(self, title, date, main_post):
        self.title = title
        self.date = date
        self.main_post = main_post


class RedditPost(object):
    """Abstract class of a reddit post with several metadata.
    """
    __metaclass__ = ABCMeta

    def __init__(self, source, post_id, date, author, score, link_id=None, parent_id=None, author_flair=None):
        self.post_id = post_id
        self.date = date
        self.author = author
        self.score = score
        self.link_id = link_id
        self.parent_id = parent_id
        self.author_flair = author_flair
        self.source = source
        self.body = None

    def get_attrib(self, save_source=False):
        attrib = {u'id': self.post_id, u'date': self.date, u'score': self.score, u'author': self.author}
        if self.link_id:
            attrib[u'link_id'] = self.link_id
        if self.parent_id:
            attrib[u'parent_id'] = self.parent_id
        if self.author_flair:
            attrib[u'author_flair'] = self.author_flair
        if save_source:
            attrib[u'source'] = self.source
        return attrib

    @abstractmethod
    def make_xml_subelement(self, parent, save_source=False):
        pass


class RedditCommentPost(RedditPost):
    """Class of a reddit comment post with a text body.
    """

    def __init__(self, source, post_id, date, author, score, body, link_id=None, parent_id=None, author_flair=None):
        super(self.__class__, self).__init__(source, post_id, date, author, score, link_id=link_id, parent_id=parent_id,
                                             author_flair=author_flair)
        self.body = body

    def make_xml_subelement(self, parent, save_source=False):
        subelement = ET.SubElement(parent, u'comment',
                                   attrib=super(self.__class__, self).get_attrib(save_source=save_source))
        subelement.text = self.body
        return subelement


class RedditLinkPost(RedditPost):
    """Class of a reddit link post with a url referencing another reddit post or external website.
    """

    def __init__(self, source, post_id, date, author, score, url, domain,
                 link_id=None, parent_id=None, author_flair=None):
        super(self.__class__, self).__init__(source, post_id, date, author, score, link_id=link_id, parent_id=parent_id,
                                             author_flair=author_flair)
        self.domain = domain
        self.url = url

    def make_xml_subelement(self, parent, save_source=False):
        subelement = ET.SubElement(parent, u'link',
                                   attrib=super(self.__class__, self).get_attrib(save_source=save_source))
        subelement.set(u'domain', self.domain)
        subelement.set(u'url', self.url)
        return subelement


def save_descendants_for_parent(potential_children, parent_element, save_source=False):
    """ Recursive function to sort a list of descendants of a given parent node into a tree structure.
    """
    index = 0
    while index < len(potential_children):
        child = potential_children[index]
        if child.parent_id == parent_element.get(u'id'):
            # create xml subelement for child
            child_element = child.make_xml_subelement(parent_element, save_source=save_source)
            # remove it from list
            potential_children.pop(index)
            # find descendants of child recursively
            save_descendants_for_parent(potential_children, child_element, save_source=save_source)
        else:
            index += 1


class SubredditCorpus(object):
    """Class for a corpus structure containing reddit submissions and posts structured in a tree.
    """

    def __init__(self, subreddit_name, subreddit_id, download_date):
        self.subreddit_name = subreddit_name
        self.subreddit_id = subreddit_id
        self.download_date = download_date
        self.submissions = {}

    def to_xml(self, save_source=False):
        corpus = ET.Element(u'subredditcorpus', attrib={u'subreddit_name': self.subreddit_name,
                                                        u'subreddit_id': self.subreddit_id,
                                                        u'download_date': self.download_date})
        sys.stderr.write('Creating tree structure of submissions and comments.\n')
        submission_index = 0
        for submission_id, submission in self.submissions.items():
            submission_index += 1
            if submission_index % 100 == 1:
                sys.stderr.write('Processed submission %d of %d\n' % (submission_index, len(self.submissions)))
            submission_element = ET.SubElement(corpus, u'submission', attrib={u'id': submission.submission_id})
            if submission.title:
                submission_element.set(u'title', submission.title)
            if submission.date:
                submission_element.set(u'date', submission.date)
            if submission.main_post is not None:
                main_post_element = ET.SubElement(submission_element, u'main_post')
                submission.main_post.make_xml_subelement(main_post_element, save_source=save_source)
            # sort comments
            save_descendants_for_parent(submission.comments, submission_element, save_source=save_source)
        return corpus


def process_download_to_xml_corpus(path, subreddit_name, subreddit_id, download_date, save_source=False):
    """ Create a subreddit corpus object and store comments and submissions from a given download folder into this
    corpus object structured as a tree.
    """

    corpus = SubredditCorpus(subreddit_name, subreddit_id, download_date)

    index = 0
    max_index = float("inf")
    file_num = 0
    for top, _, files in os.walk(path):
        if index == max_index:
            break
        total_num_files = len(files)
        for filename in files:
            file_num += 1
            if file_num % 10 == 1:
                sys.stderr.write('Reading file %d of %d.\n' % (file_num, total_num_files))
            if index == max_index:
                break

            dl_c, dl_s = pickle.load(open(os.path.join(top, filename), mode='rb'))
            posts = [('comment', post) for post in dl_c] + [('submission', post) for post in dl_s]

            for post_type, post_dict in posts:
                index += 1
                if index == max_index:
                    break

                source = str(post_dict) if save_source else None

                post_id = post_dict['id']
                date = str(post_dict['created_utc'])
                author = post_dict['author']
                score = str(post_dict['score'])

                link_id = post_dict.get('link_id', '').split('_')[-1] or None
                parent_id = post_dict.get('parent_id', '').split('_')[-1] or None
                author_flair = post_dict.get('author_flair_text', '') or None

                # check if submission is already saved
                submission_id = link_id or post_id
                if submission_id not in corpus.submissions:
                    submission = RedditSubmission(submission_id)
                    corpus.submissions[submission_id] = submission
                else:
                    submission = corpus.submissions[submission_id]

                # save post depending on post_type as submission or comment
                if post_type == 'submission':
                    # check if submission contains just link or text (or is empty/deleted)
                    try:
                        post_text = post_dict['selftext']
                        post_obj = RedditCommentPost(source, post_id, date, author, score, post_text, link_id,
                                                     parent_id, author_flair)
                    except KeyError:
                        url = post_dict['url']
                        domain = post_dict['domain']
                        post_obj = RedditLinkPost(source, post_id, date, author, score, url, domain, link_id, parent_id,
                                                  author_flair)
                    title = post_dict['title']
                    submission.add_main_post(title, date, post_obj)
                elif post_type == 'comment':
                    post_text = post_dict['body']
                    post_obj = RedditCommentPost(source, post_id, date, author, score, post_text, link_id, parent_id,
                                                 author_flair)
                    submission.comments.append(post_obj)
                else:
                    raise ValueError('Error: unknown post type "%s".' % post_type)
    sys.stderr.write('Finished reading input files.\n')
    return corpus
