from typing import Tuple, Dict

import pandas as pd
import numpy as np

__all__ = ['Preprocess']


class Preprocess:
    def _get_full_map_data(self) -> pd.DataFrame:
        df = self._get_data()
        movie_date_map, movie_title_map = self._mapper_generator()
        df['Movie_Date'] = df['Movie_Id'].map(movie_date_map).astype(str)
        df['Movie_Title'] = df['Movie_Id'].map(movie_title_map)

        return df

    def _get_data(self) -> pd.DataFrame:
        l = []
        for i in range(1, 5):
            df = self._read_raw(f'data/combined_data_{i}.txt')
            df_processed = self._preprocess(df, str(i))
            l.append(df_processed)

        return pd.concat(l)

    @staticmethod
    def _read_raw(file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, header=None, names=['Cust_Id', 'Rating', 'Review_Date'], usecols=[0, 1, 2])
        df['Rating'] = df['Rating'].astype(float)
        return df

    @staticmethod
    def _preprocess(data: pd.DataFrame, index: str) -> pd.DataFrame:
        df_nan = pd.DataFrame(pd.isnull(data.Rating))
        df_nan = df_nan[df_nan['Rating'] == True]
        df_nan = df_nan.reset_index()

        movie_np = []
        movie_id = {'1': 1, '2': 4500, '3': 9211, '4': 13368}[index]

        for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
            temp = np.full((1, i - j - 1), movie_id)
            movie_np = np.append(movie_np, temp)
            movie_id += 1

        last_record = np.full((1, len(data) - df_nan.iloc[-1, 0] - 1), movie_id)
        movie_np = np.append(movie_np, last_record)

        df = data[pd.notnull(data['Rating'])]

        df['Movie_Id'] = movie_np.astype(int)
        df['Cust_Id'] = df['Cust_Id'].astype(int)

        return df

    @staticmethod
    def _mapper_generator() -> Tuple[Dict[int, int], Dict[int, str]]:
        movies = pd.read_csv('data/movie_titles.csv', names=['Movie_Id', 'year' , 'title'], sep=',', encoding='latin1')

        movie_date_map, movie_title_map = {}, {}
        for id, year in zip(movies['Movie_Id'], movies['year']):
            movie_date_map[id] = year

        for id, title in zip(movies['Movie_Id'], movies['title']):
            movie_title_map[id] = title

        return movie_date_map, movie_title_map
