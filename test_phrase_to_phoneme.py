#!/usr/bin/env python
#-*- encoding:utf-8 -*-

from __future__ import unicode_literals

from argparse import ArgumentParser

import os
import codecs
from data_load import text_normalize
from PETRUS.g2p.g2p import G2PTranscriber
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
        dest='output',
        required=True,
        help=u'output phoneme file',
    )
    parser.add_argument(
        '-s', '--separator',
        dest='separator',
        required=False,
        default ='. ',
        help=u"separator between identify and text default: ' ' ",
    )
    args = parser.parse_args()

    # Open output file
    f = codecs.open(args.output, 'w', 'utf-8')
    # Iterate input file
    for line in args.file.readlines():
        # Get input word
        print('separator',args.separator)
        identify,line= line.split(args.separator, 1)
        line = text_normalize(line).replace(',',' , ').replace('?',' ? ')
        print('linha:',line)
        words = line.strip().lower().split(' ')
        transcrito = [] 
        for word in words:
            print(word)
            # Initialize g2p transcriber
            g2p = G2PTranscriber(word, algorithm='silva')
            # Write file
            #print(g2p.transcriber())
            try:
                transcription = g2p.transcriber()
                transcrito.append(str(transcription))
            except:
                transcrito.append(g2p.transcriber())
           
        print(identify+'. '+"_".join(transcrito))
        f.write(identify+'. '+"_".join(transcrito)+'\n')
    # Close output file
    f.close()

    print('\nSuccess!!! Open the "',args.output,'" file to see the result.\n')
