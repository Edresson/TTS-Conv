#!/usr/bin/env python
#-*- encoding:utf-8 -*-

from __future__ import unicode_literals

from argparse import ArgumentParser

from g2p.g2p import G2PTranscriber

import os
import codecs
from data_load import text_normalize
from utils import texts_to_phonemes

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("the file \"%s\" does not exist!" % arg)
    elif not arg.endswith(".txt"):
        parser.error("\"%s\" is not a text file!" % arg)
    else:
        return codecs.open(arg, 'r', 'utf-8')  # return an open file handle

if __name__ == '__main__':
    # Initialize ArgumentParser class
    parser = ArgumentParser()
    # Parse command line arguments
    parser.add_argument(
        '-f', '--file',
        dest='file',
        required=True,
        help=u'Text file',
        type=lambda x: is_valid_file(parser, x),
    )
    parser.add_argument(
        '-o', '--output',
        dest='output file',
        required=True,
        help=u'output phoneme file',
    )
    args = parser.parse_args()

    # Open output file
    f = codecs.open('output.txt', 'w', 'utf-8')
    # Iterate input file
    for line in args.file.readlines():
        # Get input word
        directory,line= line.split('==')
        words = line.strip().lower().split(' ')
        transcrito = [] 
        for word in words:
            print(word)
            # Initialize g2p transcriber
            g2p = G2PTranscriber(word, algorithm='silva')
            # Write file
            #print(g2p.transcriber())
            try:
                transcrito.append(directory+'=='+str(g2p.transcriber()[0]))
            except:
                transcrito.append(g2p.transcriber())
           
        #print(" ".join(transcrito))
        f.write("_".join(transcrito)+'\n')
    # Close output file
    f.close()

    print('\nSuccess!!! Open the "output.txt" file to see the result.\n')
