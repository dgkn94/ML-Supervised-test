import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

hava = ["Gunesli", "Yagmurlu", "Bulutlu","Bulutlu","Bulutlu","Yagmurlu","Yagmurlu","Gunesli","Yagmurlu","Bulutlu","Yagmurlu","Gunesli","Bulutlu","Yagmurlu","Gunesli"]
sicaklik = ["Sicak","Soguk","Nemli", "Nemli", "Nemli", "Soguk", "Soguk", "Sicak", "Soguk", "Nemli", "Soguk", "Sicak","Nemli","Soguk","Sicak"]
sokagaCik = ["Evet","Hayir","Evet","Evet","Evet","Hayir","Hayir","Evet","Hayir","Evet","Hayir","Evet","Evet","Hayir","Evet"]
# 0 bulut 1 gunes 2 Yagmurlu
# 0 nemli 1 sicak 2 Soguk
# 0 evet 1 hayir
encode = LabelEncoder()

hava_enc = encode.fit_transform(hava)
sicaklik_enc = encode.fit_transform(sicaklik)
sokagaCik = encode.fit_transform(sokagaCik)
print(hava_enc)
print(sicaklik_enc)
print(sokagaCik)

ozellikler = list(zip(hava_enc, sicaklik_enc))


model = KNeighborsClassifier(n_neighbors = 3)
model.fit(ozellikler,sokagaCik)


tahmin = model.predict([[1,2]])
print(tahmin)
