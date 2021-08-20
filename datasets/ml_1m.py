from .base import AbstractDataset

import pandas as pd
import numpy as np

from datetime import date


class ML1MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README',
                'movies.dat',
                'ratings.dat',
                'users.dat']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(file_path, sep=',', header=None,names=['uid', 'sid', 'rating', 'timestamp'],dtype={'timestamp':np.float64})
        # df.head()
        # df['timestamp']=np.float(df['timestamp'])
        # df.columns = ['uid', 'sid', 'rating', 'timestamp']

        # pd.read_csv('/content/drive/MyDrive/SASRec.pytorch/data/ratings_jd.dat', sep=',',dtype={'time':np.object_},header=None,names=['user','item','ratings','time'],usecols=[0,1,2,3])
        return df


