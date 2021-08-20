from .base import AbstractNegativeSampler

from tqdm import trange

from collections import Counter


class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular'

    def generate_negative_samples(self):
        popular_items = self.items_by_popularity()

        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.user_count):
            seen = set(self.train[user[0]])
            seen.update(self.val[user][0])
            seen.update(self.test[user][0])

            samples = []
            for item in popular_items:
                if len(samples) == self.sample_size:
                    break
                if item in seen:
                    continue
                samples.append(item)

            negative_samples[user] = samples

        return negative_samples

    def items_by_popularity(self):
        popularity = Counter()
        for user in range(self.user_count):
            popularity.update(self.train[user][0])
            popularity.update(self.val[user][0])
            popularity.update(self.test[user][0])
        popular_items = sorted(popularity, key=popularity.get, reverse=True)
        return popular_items
