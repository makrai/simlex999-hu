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
        lg_fm = "%(module)s (%(lineno)s) %(levelname)s %(message)s"
        "%(asctime)s"
        self.parse_args()
        logging.basicConfig(format=lg_fm, level=logging.INFO)
        #, filename=self.args.log_filen, filemode='w')
        self.dicts = [self.read_dict(fn) for fn in self.args.dicts]
        self.embed = self.get_embed(self.args.embed)
        self.en_hu = defaultdict(set)
        self.oov_dict, self.oov_embed, self.oov_synon = set(), set(), set()

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
        parser.add_argument(
            '--dicts', nargs='+', help="dictionary tsv's in order of reliability",
            default=['/mnt/store/makrai/data/language/hungarian/dict/wikt2dict-en-hu'])
        parser.add_argument(
            '--embed', help='without extension',
            default='/mnt/permanent/Language/Hungarian/Embed/webkorp/word2vec_n5_152_m50_w10_0l_0#')
        parser.add_argument('--oov_dict', default='oov_dict.log')
        parser.add_argument('--oov_embed', default='oov_embed.log')
        parser.add_argument('--oov_synon', default='oov_synon.log')
            #webkorp/glove-hu_152')
            #webkorp/webkorp_d600_w12_m250_cbow_h0_n7_i12')
            #webkorp/webkorp_d600_w12_m250_cbow_h0_n7_i12')
            #mnsz2_webcorp/word2vec-mnsz2-webcorp_600_w10_n5_i1_m10')
            #word2vec-mnsz2-webcorp_300_w5_s0_hs0_n5_i1_m5_sgram')
        parser.add_argument('--output_filen', default='out.tsd')
        parser.add_argument('--log_filen', default='log.log')
        self.args = parser.parse_args()

    def main(self):
        with open(self.args.simlex) as simlex_f, open(
                self.args.output_filen, mode='w') as out_f:
            logging.info('reading {}, writing to {}'.format(
                self.args.simlex, self.args.output_filen))
            for line in simlex_f:
                en1, en2, _pos, sim, tail = line.strip().split('\t', 4)
                logging.debug((en1, en2))
                if en1 == 'word1':
                    logging.info('skipping header {}'.format(line))
                    continue
                try:
                    hus1, hus2 = [self.translate(word) for word in [en1, en2]]
                    hu1, hu2, len_intersection = self.disambig(en1, en2, hus1,
                                                               hus2, sim)
                except OOVException as e:
                    logging.debug((e, en1, en2, hus1, hus2))
                    continue                    
                self.en_hu[en1].add(hu1)
                self.en_hu[en2].add(hu2)
                logging.debug((hu1, hu2))
                out_f.write('{}\t{}\t{}\t{}\t{}\n'.format(
                    en1, en2, hu1, hu2, len_intersection))
        self.logg_ambig()
        self.logg_oov(self.oov_dict, self.args.oov_dict)
        self.logg_oov(self.oov_embed, self.args.oov_embed)
        self.logg_oov(self.oov_synon, self.args.oov_synon)

    def translate(self, word):
        for dict_ in self.dicts:
            if word in dict_:
                return dict_[word]
        self.oov_dict.add(word)
        raise(OOVException('not in dictionary: {}'.format(word)))

    def disambig(self, en1, en2, hus1, hus2, sim):
        # Different disambiguation strategy is followed depending on the size
        # of the intersection of the dictionary-provided translations of the
        # two words in the similarity pair.
        intersection = hus1.intersection(hus2)
        len_intersection = len(intersection)
        if len_intersection == 2: 
            return list(intersection) + [len_intersection]
        elif len_intersection == 0:
            hus1, hus2 = (list(set_) for set_ in [hus1, hus2])
        elif len_intersection == 1:
            hus1, hus2 = (list(intersection),
                          list(hus1.symmetric_difference(hus2)))
            if not hus2:
                self.oov_synon.add((en1, en2)) 
                raise OOVException(
                    'The only translation for both {} is {}'.format(
                        (en1, en2), hus1))
        elif len_intersection > 2:
            raise NotImplementedError('Three synonyms: {}'.intersection)
        mx1, mx2 = (self.words2mx(words) for words in [hus1, hus2])
        w1_ids, w2_ids = zip(*self.argclosest(mx1, mx2, sim))
        logging.debug(np.array(hus1)[np.array(w1_ids)])
        logging.debug(np.array(hus2)[np.array(w2_ids)])
        return hus1[w1_ids[0]], hus2[w2_ids[0]], len_intersection

    def words2mx(self, words):
        mx = np.asarray([self.embed[word] 
                         for word in words if word in self.embed])
        if len(mx):
            return mx
        else:
            for word in words:
                self.oov_embed.add(word) 
            raise OOVException('not in embedding: {}'.format(
                ' '.join(words)))

    def argclosest(self, mx1, mx2, sim=10):
        product = np.absolute(mx1.dot(mx2.T) - float(sim)/10)
        elems0 = np.argsort(product, axis=None)
        return [np.unravel_index(elem0, product.shape) for elem0 in elems0]

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

    def logg_ambig(self):
        """
        This function logs the translations we used if there are more.
        """
        for en, hus in self.en_hu.items():
            # sorted(self.en_hu, key=lambda en: len(self.en_hu[en]),
            # reverse=True):
            if len(hus) != 1:
                logging.debug((en, hus))

    def logg_oov(self, set_, filen):
        with open(filen, mode='w') as log_f:
            for item in set_:
                log_f.write('{}\n'.format(item))

class OOVException(Exception):
    pass

if __name__ == '__main__':
    SimLex999Translator().main()
