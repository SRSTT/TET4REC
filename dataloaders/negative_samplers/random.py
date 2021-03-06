from .base import AbstractNegativeSampler

from tqdm import trange

import numpy as np


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        self.seed=10
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.user_count):
            if isinstance(self.train[user][1][0], tuple):
                seen = set(x[0] for x in self.train[user][0])
                seen.update(x[0] for x in self.val[user][0])
                seen.update(x[0] for x in self.test[user][0])
            else:
                seen = set(self.train[user][0])
                seen.update(self.val[user][0])
                seen.update(self.test[user][0])

            samples = []
            for _ in range(self.sample_size):
                item = np.random.choice(self.item_count) + 1
                while item in seen or item in samples:
                    item = np.random.choice(self.item_count) + 1
                samples.append(item)

            negative_samples[user] = samples

        return negative_samples
