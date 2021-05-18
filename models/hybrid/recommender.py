from typing import List, Dict

import numpy as np
import pandas as pd
import pickle

from cached_property import cached_property
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k

from models.hybrid.base import *

__all__ = ['LightFMRecommender']


class LightFMRecommender(Base):
    def run(self, epochs: int = 1, no_components: int = 50, learning_rate:float = 0.05) -> Dict[str, float]:
        """
         build interaction matrix -> build movie features -> build model

        Example (5000 samples, 50 components, 5 epochs, learning_rate=0.05)
        =================================
        {'auc_train': 0.66268414, 'auc_test': 0.67257625,
         'precision_train@10': 0.035984848, 'precision_test@10': 0.014193548,
         'recall_train@10': 0.06827082513973247, 'recall_test@10': 0.0646373101211811}

        ###########################
        #### Random Stratified ####
        ###########################
        Example (2 million samples, 50 components, 1 epochs, learning_rate=0.05)
        =================================
        {'auc_train': 0.5171841, 'auc_test': 0.51610065,
         'precision_train@10': 0.018248174, 'precision_test@10': 0.0040145987,
         'recall_train@10': 0.0008001067196610589, 'recall_t0.018248174est@10': 0.0007001527280332769}

        ########################
        #### Popular Active ####
        ########################
        Example (333000 samples, 150 components, 1 epochs, learning_rate=0.05)  20% test data
        =================================
        {'auc_train': 0.63388383, 'auc_test': 0.5569484,
        'precision_train@10': 0.7255412, 'precision_test@10': 0.17099567,
        'recall_train@10': 0.006322884137545113, 'recall_test@10': 0.006053869700910709}

        Example (333000 samples, 50 components, 1 epochs, learning_rate=0.05)  40% test data
        =================================
        {'auc_train': 0.6001097, 'auc_test': 0.56429684,
         'precision_train@10': 0.56060606, 'precision_test@10': 0.33030304,
         'recall_train@10': 0.006517918240037026, 'recall_test@10': 0.005792534657980192}

        Example (333000 samples, 50 components, 20 epochs, learning_rate=0.05)  40% test data
        =================================
        {'auc_train': 0.6077434, 'auc_test': 0.5688331,
         'precision_train@10': 0.5874459, 'precision_test@10': 0.32424247,
         'recall_train@10': 0.0068082500065638684, 'recall_test@10': 0.005756504594433489}

        Example (333000 samples, 50 components, 1 epochs, learning_rate=0.05)  40% test data with normalization
        =================================
        {'auc_train': 0.60080063, 'auc_test': 0.56425303,
         'precision_train@10': 0.56926405, 'precision_test@10': 0.33679655,
         'recall_train@10': 0.006628036812872702, 'recall_test@10': 0.005913302996971047}
         """
        ## Build Matrix Factorization between Customer and Movie
        data = self._filter_data

        dataset = Dataset()
        dataset.fit(data['Cust_Id'].unique(), data['Movie_Id'].unique(), item_features=self.get_combination)
        (interactions, weights) = dataset.build_interactions([(x['Cust_Id'], x['Movie_Id'], x['Rating'])
                                                              for index, x in data.iterrows()])

        train, test = random_train_test_split(interactions, test_percentage=0.4,
                                              random_state=np.random.RandomState(7))
        print("Finished creating interactions matrix!")

        ## Build movie features
        movies_id, tfidf_data = self.get_tfidf_matrix
        features_lists = [list(x) for x in tfidf_data.values]
        movies_features = dataset.build_item_features(data=self.get_movies_tuple(features_lists, movies_id, tfidf_data),
                                                      normalize=True)
        print("Finished building movie features!")

        ## Build model
        model = LightFM(no_components=no_components, learning_rate=learning_rate, loss='warp', k=15)
        model.fit(train, epochs=epochs, item_features=movies_features, num_threads=4)
        print("Finished building LightFM model!")

        with open('hybrid_model_popular_active.pickle', 'wb') as fle:
            pickle.dump(model, fle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Finished saving LightFM model!")

        return {
            "auc_train": auc_score(model, train, item_features=movies_features).mean(),
            "auc_test": auc_score(model, test, item_features=movies_features).mean(),
            "precision_train@10": precision_at_k(model, train, item_features=movies_features, k=10).mean(),
            "precision_test@10": precision_at_k(model, test, item_features=movies_features, k=10).mean(),
            "recall_train@10": recall_at_k(model, train, item_features=movies_features, k=10).mean(),
            "recall_test@10": recall_at_k(model, test, item_features=movies_features, k=10).mean()
        }

    def get_movies_tuple(self, features_lists: List[List[float]], movies_id: List[str], tfidf_data: pd.DataFrame):
        tmp = [self._format_features(item, tfidf_data) for item in features_lists]

        return list(zip(movies_id, tmp))

    @cached_property
    def get_combination(self) -> List[str]:
        """ Get combination of features name and features value to be fitted into LightFM """
        movies_id, tfidf_data = self.get_tfidf_matrix

        l = []
        for col in tfidf_data:
            l.extend([(f'{col}:' + str(i)) for i in tfidf_data[col]])

        return list(set(l))

    @staticmethod
    def _format_features(item: List[float], tfidf_data: pd.DataFrame):
        result = []
        formated_column_names = [str(col) + ':' for col in tfidf_data.columns]
        for x, y in zip(formated_column_names, item):
            res = str(x) + str(y)
            result.append(res)

        return result
