from typing import List, Tuple
from pathlib import Path

import pandas as pd
import re
import string

from cached_property import cached_property
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


__all__ = ['Base', "TextCleaner"]


class Base:
    @cached_property
    def get_tfidf_matrix(self) -> Tuple[List[str], pd.DataFrame]:
        data = self._read('movie_titles')
        data = data[data['Movie_Id'].isin(self._filter_data['Movie_Id'].tolist())]
        TextCleaner.transform(data, 'title')

        tfidf = TfidfVectorizer()

        movies_id = data.Movie_Id.tolist()
        tfidf_matrix = tfidf.fit_transform(data['title'])

        return movies_id, pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names())

    @cached_property
    def _filter_data(self):
        """ To reduce size of sparse matrix during matrix factorization approach """
        min_movie_ratings = 10000 #0.885 quantile
        min_user_ratings = 3000 #0.97 quantile

        df = self._read('final')
        filter_movies = (df['Movie_Id'].value_counts() > min_movie_ratings)
        filter_movies = filter_movies[filter_movies].index.tolist()

        filter_users = (df['Cust_Id'].value_counts() > min_user_ratings)
        filter_users = filter_users[filter_users].index.tolist()

        return df[(df['Movie_Id'].isin(filter_movies)) & (df['Cust_Id'].isin(filter_users))]

    @cached_property
    def _train_test_split(self):
        df_train, df_test = train_test_split(self._filter_data, test_size=0.2, random_state=44)
        return df_train, df_test

    @cached_property
    def get_matrix_factorized_data(self):
        df_train, df_test = self._train_test_split
        return df_train.pivot_table(index='Cust_Id', columns='Movie_Id', values='Rating')

    @staticmethod
    def _read(filename: str) -> pd.DataFrame:
        path = Path(__file__).parents[1].joinpath("data").as_posix()
        if filename == 'movie_titles':
            return pd.read_csv(f'{path}/{filename}.csv',
                               names=['Movie_Id', 'year' , 'title'], sep=',', encoding='latin1')
        return pd.read_csv(f'{path}/{filename}.csv')


class TextCleaner:
    stop_word = stopwords.words('english')

    @classmethod
    def transform(cls, data: pd.DataFrame, column: str, lemma: bool = False, stemming: bool = False):
        tokenizer = RegexpTokenizer(r'\w+')

        # remove punctuation
        data[column] = data[column].apply(lambda x: re.sub(r'[%s]' % re.escape(string.punctuation), '', x))

        # remove digits
        data[column] = data[column].apply(lambda x: re.sub(r'\w*\d\w*', '', x))

        # tokenization -> stopwords_removal -> lemmatization -> stemming -> concatenating words
        data[column] = data[column].apply(lambda x: tokenizer.tokenize(x.lower()))
        data[column] = data[column].apply(lambda x: cls._remove_stopwords(x))
        data[column] = data[column].apply(lambda x: [i for i in x if len(x) > 2])

        if lemma:
            data[column] = data[column].apply(lambda x: cls._word_lemmatizer(x))
        if stemming:
            data[column] = data[column].apply(lambda x: cls._word_stemmer(x))

        data[column] = data[column].apply(lambda x: cls._join(x))

    @classmethod
    def _remove_stopwords(cls, text) -> List:
        words = [w for w in text if w not in cls.stop_word]
        return words

    @staticmethod
    def _join(text) -> str:
        join_text = " ".join(i for i in text)
        return join_text

    @staticmethod
    def _word_lemmatizer(text) -> List:
        lemmatizer = WordNetLemmatizer()
        lem_text = [lemmatizer.lemmatize(i) for i in text]
        return lem_text

    @staticmethod
    def _word_stemmer(text) -> List:
        stemmer = PorterStemmer()
        stem_text = [stemmer.stem(i) for i in text]
        return stem_text
