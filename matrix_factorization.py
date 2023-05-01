

#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################
from typing import Union, Any

# !pip install surprise
import pandas as pd
import numpy as np

from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

#---------------------------------------------------
from collections import defaultdict
import os
import io
#---------------------------------------------------

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

sample_df.shape

user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                          values="rating")

user_movie_df.shape

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(sample_df[['userId',
                                           'movieId',
                                       'rating']], reader)

##############################
# Adım 2: Modelleme
##############################

trainset, testset = train_test_split(data, test_size=.25)
svd_model = SVD()
svd_model.fit(trainset)
predictions = svd_model.test(testset)

#------------------------------------------------------
def hata_k_o_k(tahminler):
    h_k_Ort = np.mean(
        [float((g_deger - tahmin) ** 2) for (_, _, g_deger, tahmin, _) in tahminler]
    )
    h_k_o_Karekok = np.sqrt(h_k_Ort)

    return h_k_o_Karekok
#------------------------------------------------------

hata_k_o_k(predictions)

svd_model.predict(uid=4.0, iid=356, verbose=True)
sample_df[sample_df["userId"] == 4]

##############################
# Adım 3: Model Tuning
##############################

param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}

gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)
gs.fit(data)

gs.best_score['rmse']
gs.best_params['rmse']

accuracy
#0.9419
##############################
# Adım 4: Final Model ve Tahmin
##############################

dir(svd_model)
svd_model.n_epochs

#svd_model = SVD(**gs.best_params['hataKarelerOrtalamasiKarekok'])

svd_model = SVD(**gs.best_params['rmse'])

#n_epochs =  10, lr_all = 0.002
data = data.build_full_trainset()
svd_model.fit(data)

svd_model.predict(uid=4.0, iid=356, verbose=True)
sample_df[sample_df["userId"] == 4]

#------------------------------------------------------
def tavsiyeYap(predictions,n = 10):
    en_iyi_10 = defaultdict(list)
    for uid, iid, g_deger, tahmin, _ in predictions:
        en_iyi_10[uid].append((iid, tahmin))

    for uid, user_ratings in en_iyi_10.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        #azalan sırada sıralar
        en_iyi_10[uid] = user_ratings[:n]

    return en_iyi_10
#------------------------------------------------------

a = tavsiyeYap(predictions,10)
sıralı_tavsiye = {k:v for k, v in sorted(a.items())}

####################################################


from collections import defaultdict
import os
import io

def tavsiyeYap(predictions,n = 10):
    en_iyi_10 = defaultdict(list)
    for uid, iid, g_deger, tahmin, _ in predictions:
        en_iyi_10[uid].append((iid, tahmin))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        #azalan sırada sıralar
        en_iyi_10[uid] = user_ratings[:n]

    return en_iyi_10

def hata_k_o_k(tahminler):
    h_k_Ort = np.mean(
        [float((g_deger - tahmin) ** 2) for (_, _, g_deger, tahmin, _) in tahminler]
    )
    h_k_o_Karekok = np.sqrt(h_k_Ort)

    return h_k_o_Karekok

def hata_k_o(tahminler):
    h_k_Ort = np.mean(
        [float((g_deger - tahmin) ** 2) for (_, _, g_deger, tahmin, _) in tahminler]
    )

    return h_k_Ort

def ortalama_m_h(tahminler):
    ort_mut_Hata = np.mean(
        [float(abs(g_deger - tahmin)) for (_, _, g_deger, tahmin, _) in tahminler]
    )

    return ort_mut_Hata

