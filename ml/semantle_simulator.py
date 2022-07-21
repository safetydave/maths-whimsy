from gensim.models import KeyedVectors


class SemantleSimulator:

    def __init__(self):
        semantle_model_ref = "data/GoogleNews-vectors-negative300.bin"
        self.wv = KeyedVectors.load_word2vec_format(semantle_model_ref, binary=True)
        self.score_scaling = 100
        self.target = None

    def score_guess(self, guess, target=None):
        word = self.target if target is None else target
        score = 0
        if guess in self.wv:
            score = self.wv.similarity(word, guess) * self.score_scaling
        return score

    # todo I think semantle excludes uncommon words in this ranking
    def score_calibration(self, target=None):
        word = self.target if target is None else target
        sbw = self.wv.similar_by_word(word, topn=1000)
        sbw_scaled = [(s0, s1 * self.score_scaling) for s0, s1 in sbw]
        return sbw_scaled[0], sbw_scaled[9], sbw_scaled[999]
