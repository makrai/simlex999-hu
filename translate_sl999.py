import argparse
from collections import defaultdict
from io import open
import logging
import os

import numpy as np
from gensim.models import KeyedVectors

class SimLex999Translator():
    # What if a word would be disambiguated differently in different pairs
    # containing it?
    def __init__(self):
        lg_fm = "%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s"
        self.parse_args()
        logging.basicConfig(format=lg_fm, level=logging.INFO,
                            filename=self.args.log_filen, filemode='w')
        self.en_hu = {}

    def main(self):
        self.dicts = [self.read_dict(fn) for fn in self.args.dicts]
        self.embed = self.get_embed(self.args.embed)
        self.en_hu = {}
        # en_hu is for checking whether multiple occurrences of a word are
        # disambiguated the same way
        with open(self.args.simlex) as f, open(self.args.output_filen,
                                               mode='w') as out_f:
            logging.info('reading {}, writing to {}'.format(
                self.args.simlex, self.args.output_filen))
            for line in f:
                en1, en2, tail = line.strip().split('\t', 2)
                logging.info((en1, en2))
                if en1 == 'word1':
                    logging.info('skipping header {}'.format(line))
                    continue
                try:
                    hus1, hus2 = [self.translate(word) for word in [en1, en2]]
                except OOVException as e:
                    logging.warn(e)
                    continue                    
                intersection = hus1.intersection(hus2)
                if intersection and len(intersection) == 2: 
                    hu1, hu2 = intersection
                else:
                    if not intersection:
                        hus1, hus2 = (list(set_) for set_ in [hus1, hus2])
                    elif len(intersection) == 1:
                        hus1, hus2 = (list(intersection),
                                      list(hus1.symmetric_difference(hus2)))
                    elif len(intersection) > 2:
                        raise logging.warn('Three synonyms: {}'.intersection)
                    try:
                        mx1, mx2 = (self.words2mx(words) 
                                    for words in [hus1, hus2])
                    except OOVException as e:
                        logging.warn(e)
                        continue
                    w1_ids, w2_ids = zip(*self.argclosest(mx1, mx2))
                    logging.debug(np.array(hus1)[np.array(w1_ids)])
                    logging.debug(np.array(hus2)[np.array(w2_ids)])
                    hu1, hu2 = hus1[w1_ids[0]], hus2[w2_ids[0]]
                logging.info((hu1, hu2))
                out_f.write('{}\t{}\t{}\n'.format(hu1, hu2, tail))

    def argclosest(self, mx1, mx2):
        product = mx1.dot(mx2.T)
        elems0 = np.argsort(-product, axis=None)
        return [np.unravel_index(elem0, product.shape) for elem0 in elems0]

    def translate(self, word):
        for dict_ in self.dicts:
            if word in dict_:
                return dict_[word]
        raise(OOVException('oov: {} {}'.format(word, len(word))))

    def words2mx(self, words):
        mx = np.asarray([self.embed[word] 
                         for word in words if word in self.embed])
        if len(mx):
            return mx
        else:
            raise OOVException('none of words {} found in embedding'.format(
                ' '.join(words)))

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description='Translates a tipically English word similarity dataset \
            like SimLex999 to some other language using a list of dictionaries \
            and a word embedding.  Such datasets contain word pair with a \
            measure of semantic similarity.  Dictionaries should be specified \
            in order of reliability, and may contain more translational \
            equivalents for ambiguous words. In that case, we choose the \
            translation wich is most similar to the other word in the semantic \
            similarity pair.')
        parser.add_argument(
            '--simlex', help='input word similarity dataset',
            default = '/mnt/permanent/Language/English/Data/SimLex-999/SimLex-999.txt')
        parser.add_argument('dicts', nargs='+', help="dictionary tsv's in order \
                            of reliability")
        parser.add_argument(
            '--embed', help='without extension',
            default='/mnt/permanent/Language/Hungarian/Embed/webkorp/word2vec_n5_152_m50_w10_0l_0#')
            #webkorp/glove-hu_152')
            #webkorp/webkorp_d600_w12_m250_cbow_h0_n7_i12')
            #webkorp/webkorp_d600_w12_m250_cbow_h0_n7_i12')
            #mnsz2_webcorp/word2vec-mnsz2-webcorp_600_w10_n5_i1_m10')
            #word2vec-mnsz2-webcorp_300_w5_s0_hs0_n5_i1_m5_sgram')
        parser.add_argument('--output_filen', default='out.tsd')
        parser.add_argument('--log_filen', default='log.log')
        self.args = parser.parse_args()

    def read_dict(self, dict_fn):
        dict_ = defaultdict(set)
        with open(dict_fn) as dict_f: #, encoding="utf-8"
            for line in dict_f:
                if ' ' in line:
                    continue
                en, hu = line.strip().split('\t')
                dict_[en].add(hu)
        return dict_

    def get_embed(self, filen):
        gens_fn = '{}.gensim'.format(filen)
        if os.path.isfile(gens_fn):
            return KeyedVectors.load(gens_fn)
        else:
            embed = KeyedVectors.load_word2vec_format('{}.w2v'.format(filen))
            embed.save(gens_fn)
            return embed

class OOVException(Exception):
    pass

if __name__ == '__main__':
    SimLex999Translator().main()
