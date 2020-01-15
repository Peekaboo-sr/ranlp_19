#!/usr/bin/python3

"""
@author Johannes Schäfer - IwiSt, University of Hildesheim

Contains functions to download reddit data using the Pushshift API in psaw.
"""

import sys
import os
import pickle
import datetime as dt
from psaw import PushshiftAPI


YEARS = (2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019)
POST_LIMIT = 1000000


def download_subreddit(subreddit_name, out_folder):
    """Function to crawl all submissions and comments of a subreddit and pickle their data and metadata.
    This function creates one pickle file for each day on which users have posted submissions or comments.
    The POST_LIMIT constant might have to be lowered for weaker systems.

    :param subreddit_name: name of the subreddit to be crawled
    :param out_folder: output folder for the pickled objects
    """
    api = PushshiftAPI()

    limit_exceeded = []

    for year in YEARS:
        for month in range(1, 13):
            for day in range(1, 32):
                # set start and end epoch
                try:
                    start_epoch = int(dt.datetime(year, month, day).timestamp())
                except ValueError:
                    # month has only 30 or less days, continue with next month
                    continue

                try:
                    end_epoch = int(dt.datetime(year, month, day+1).timestamp())
                except ValueError:
                    # end of month reached, start with first day of next month
                    try:
                        end_epoch = int(dt.datetime(year, month+1, 1).timestamp())
                    except ValueError:
                        # end of year reached, start with first month of next year
                        end_epoch = int(dt.datetime(year+1, 1, 1).timestamp())

                # query comments from the subreddit for the given epoch
                gen = api.search_comments(after=start_epoch, before=end_epoch,
                                          subreddit=subreddit_name, limit=POST_LIMIT)
                # select the post dictionary containing the metadata and text of the post
                comments = [comment[-1] for comment in list(gen)]

                # query submissions from the subreddit for the given epoch
                s_gen = api.search_submissions(after=start_epoch, before=end_epoch,
                                               subreddit=subreddit_name, limit=POST_LIMIT)
                # select the post dictionary containing the metadata and text of the post
                submissions = [subm[-1] for subm in list(s_gen)]

                if comments or submissions:
                    # save the results on hdd
                    out_filename = str(start_epoch) + '_' + str(end_epoch) + '_reddit-r-' + subreddit_name + '.pickle'
                    pickle.dump((comments, submissions),
                                open(os.path.join(out_folder + out_filename), mode='wb'))

                    sys.stderr.write('Processed queries for "%s" - "%s"; results: %5d comments, %5d submissions\n' % (
                        dt.datetime.utcfromtimestamp(start_epoch).strftime('%Y-%m-%d %H:%M:%S'),
                        dt.datetime.utcfromtimestamp(end_epoch).strftime('%Y-%m-%d %H:%M:%S'),
                        len(comments), len(submissions)))

                    # check if limit reached/exceeded
                    if (len(comments) >= POST_LIMIT) or (len(submissions) >= POST_LIMIT):
                        limit_exceeded.append((start_epoch, end_epoch))

    if limit_exceeded:
        with open(os.path.join(out_folder, 'ERRORS'), 'w') as errors_file:
            sys.stderr.write('\nErrors in %d epochs: post limit reached/exceeded (see file %s).\n'
                             % (len(limit_exceeded), os.path.join(out_folder, 'ERRORS')))
            errors_file.write('\n'.join([str(epoch[0]) + '\t' + str(epoch[1]) for epoch in limit_exceeded]) + '\n')
    else:
        sys.stderr.write('\nNo errors reported.\n')
