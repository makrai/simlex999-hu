import argparse
import logging
from collections import defaultdict

from gensim.models import KeyedVectors

class SimLex999Translator():
    # What if a word would be disambiguated differently in different pairs
    # containing it?
    def __init__(self):
        logging.basicConfig(
            format="%(asctime)s: (%(lineno)s) %(levelname)s %(message)s") 
        self.parse_args()
        self.en_hu = {}

    def __main__(self):
        self.dicts = map(self.read_dict, self.args.dicts)
        self.embed = self.get_embed(self.args.embed)
        self.en_hu = {} 
        # en_hu is for checking whether word are disambiguated the same way any
        # time
        with open(self.args.simlex) as f:
            for line in f:
                w1, w2, tail = f.split('\t', 2)


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
            'simlex', help='input word similarity dataset', 
            default = '/mnt/permanent/Language/English/Data/SimLex-999/SimLex-999.txt')
        parser.add_argument('dicts', type=list, help="dictionary tsv's in order \
                            of reliability")
        parser.add_argument(
            'embed', help='without extension',
            default='/mnt/permanent/Language/Hungarian/Embed/mnsz2_webcorp/word2vec-mnsz2-webcorp_300_w5_s0_hs0_n5_i1_m5_sgram')
        self.args = parser.parse_args() 

    def read_dict(self, dict_fn):
        dict_ = defaultdict(set)
        with open(dict_fn) as dict_f:
            for line in dict_f:
                en, hu = line.strip().split('\t')
                dict_[en].add(hu)
        return dict_ 

    def get_embed(filen):
        gens_fn = '{}.gensim'.format(filen)
        if os.path.isfile(gens_fn):
            return KeyedVectors.load(gens_fn)
        else:
            embed = KeyedVectors.load_word2vec_format('{}.w2v'.format(filen))
            embed.save(gens_fn)
            return embed
