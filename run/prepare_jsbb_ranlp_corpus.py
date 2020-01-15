#!/usr/bin/python3

"""
@author Johannes Schäfer - IwiSt, University of Hildesheim
"""

import os
import pickle
from datetime import datetime
from data import download_reddit_data
from data import reformat_dl_into_xml
from data import write
from data import linear_dialogues
from olpred import cnn_olpred


# ################################# #
# ### Required additional files ### #
# ################################# #
INPUT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
OLID_TRAIN = os.path.join(INPUT_FOLDER, 'olid-training-v1.0.tsv')
EMBEDDINGS_EN = os.path.join(INPUT_FOLDER, 'en_tweets_Dim200')
ENGLISH_WORDS = os.path.join(INPUT_FOLDER, 'brown-words_freq-greater-than-5.txt')
# ################################ #
# ### Selection of corpus data ### #
# ################################ #
SUBREDDIT_NAME = 'europe'
SUBREDDIT_ID = '2qh4j'
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../europe_corpus/')
# ########################################### #
# ### Selection of functions to be called ### #
# ########################################### #
download_data = False
reformat_download = False
predict_ol = False
write_annotated_corpus = False
extract_linear_dialogues = False
analyse_offence_in_dialogues = False
# ########################################### #


def reformat_download_to_xml():
    europe_corpus = reformat_dl_into_xml.process_download_to_xml_corpus(
        path=OUTPUT_FOLDER, subreddit_name=SUBREDDIT_NAME, subreddit_id=SUBREDDIT_ID, download_date=datetime.now())
    reformat_dl_into_xml.write_xml(os.path.join(OUTPUT_FOLDER,
                                                'reddit-r-' + SUBREDDIT_NAME + '-corpus.xml'), europe_corpus.to_xml())


if __name__ == '__main__':

    # step 1: download
    if download_data:
        download_reddit_data.download_subreddit(subreddit_name=SUBREDDIT_NAME, out_folder=OUTPUT_FOLDER)

    # step 2: reformat
    if reformat_download:
        reformat_download_to_xml()

    # step 3: predict offensive language
    anno_data_pickle_file_path = os.path.join(OUTPUT_FOLDER, 'anno_data.p')
    reddit_data_xml_corpus_filename = os.path.join(OUTPUT_FOLDER, 'redditcorpus.xml')

    if predict_ol:
        olid_data_filename = os.path.join(OUTPUT_FOLDER, OLID_TRAIN)
        word_embeddings_path = os.path.join(OUTPUT_FOLDER, EMBEDDINGS_EN)
        pickle_file_path = os.path.join(OUTPUT_FOLDER, 'data.p')

        predictions = cnn_olpred.predict_offensive_language(
            olid_data_filename, reddit_data_xml_corpus_filename, word_embeddings_path, pickle_file_path)

        with open(anno_data_pickle_file_path, mode='wb') as outfile:
            pickle.dump(predictions, outfile)

    if write_annotated_corpus:
        out_corpus_path = os.path.join(OUTPUT_FOLDER, 'annotated-corpus.xml')
        subreddit_corpus_dtd = os.path.join(OUTPUT_FOLDER, 'subredditcorpus.dtd')

        predictions = pickle.load(open(anno_data_pickle_file_path, mode='rb'))
        write.write_reddit_xml_corpus(out_corpus_path, reddit_data_xml_corpus_filename, predictions,
                                      subreddit_corpus_dtd)

    # step 4: extract linear dialogues
    if extract_linear_dialogues:
        annotated_corpus_path = os.path.join(OUTPUT_FOLDER, 'annotated-corpus.xml')
        english_words_path = os.path.join(OUTPUT_FOLDER, ENGLISH_WORDS)
        output_ld_path = os.path.join(OUTPUT_FOLDER, 'extracted_linear-dialogues.json')
        linear_dialogues.extract_linear_dialogues_from_corpus(
            annotated_corpus_path, output_ld_path, language_filter_path=english_words_path, window_size=3)

