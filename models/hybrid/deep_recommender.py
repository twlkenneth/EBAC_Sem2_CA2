from cached_property import cached_property
from typing import Tuple

import numpy as np
import pandas as pd
from numpy import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import Input, Embedding, Reshape, Concatenate, Dense, Dropout
from keras.models import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.hybrid.base import *

__all__ = ['DeepRecommender']


class DeepRecommender(Base):
    def evaluate_model(self, epochs: int = 2, batch_size: int = 1024):
        data = self._read('sampled')
        df_train, df_test, train_tf_idf, test_tf_idf = self._get_preprocessed_inputs

        user_embed = 10
        movie_embed = 10

        user_id_input = Input(shape=[1], name='user')
        movie_id_input = Input(shape=[1], name='movie')
        tfidf_input = Input(shape=(7567,), name='tfidf', sparse=True)

        user_embedding = Embedding(output_dim=user_embed,
                                   input_dim=len(data['Cust_Id']),
                                   input_length=1,
                                   name='user_embedding')(user_id_input)
        movie_embedding = Embedding(output_dim=movie_embed,
                                    input_dim=len(data['Movie_Id']),
                                    input_length=1,
                                    name='movie_embedding')(movie_id_input)

        tfidf_vectors = Dense(128, activation='relu')(tfidf_input)
        tfidf_vectors = Dense(32, activation='relu')(tfidf_vectors)

        user_vectors = Reshape([user_embed])(user_embedding)
        movie_vectors = Reshape([movie_embed])(movie_embedding)

        # Concatenate all layers into one vector
        both = Concatenate()([user_vectors, movie_vectors, tfidf_vectors])

        dense = Dense(512, activation='relu')(both)
        dense = Dropout(0.2)(dense)
        output = Dense(1)(dense)

        model = Model(inputs=[user_id_input, movie_id_input, tfidf_input], outputs=output)
        model.compile(loss='mse', optimizer='adam')

        model.fit([df_train['Cust_Id'], df_train['Movie_Id'], train_tf_idf],
                  df_train['Rating'],
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.1,
                  shuffle=True)

        model.save("deep_recommender.h5")

        y_pred = model.predict([df_test['Cust_Id'], df_test['Movie_Id'], test_tf_idf])
        y_true = df_test['Rating'].values

        return {
            'rmse': np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true)),
            'mae': mean_absolute_error(y_pred=y_pred, y_true=y_true)
        }

    @cached_property
    def _get_preprocessed_inputs(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        df_train, df_test = self._train_test_split
        tfidf_data = self._get_tfidf_matrix

        train_tf_idf = self._get_tfidf_list(df_train, tfidf_data)
        test_tf_idf = self._get_tfidf_list(df_test, tfidf_data)

        return df_train, df_test, train_tf_idf, test_tf_idf

    @staticmethod
    def _get_tfidf_list(data: pd.DataFrame, tfidf_data) -> np.ndarray:
        """ Filtered TF-IDF matrix according to training or testing  """
        # Iterate over all movie-ids and save the tfidf-vector
        tmp = [tfidf_data.iloc[id-1] for id in data['Movie_Id'].values]
        return vstack(tmp)

    @cached_property
    def _get_tfidf_matrix(self) -> pd.DataFrame:
        """ Return TF-IDF sparse matrix """
        data = self._read('movie_titles')
        TextCleaner.transform(data, 'title')
        tfidf = TfidfVectorizer()

        tfidf_matrix = tfidf.fit_transform(data['title'])

        return pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names())
