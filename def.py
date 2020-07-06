from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
print("Kutuphane basariliyla yuklendi")

# DATASETI URL'DEN CEK VE CVS HALINDE OKU
try:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    isim = ["sepal-uzunluk", "sepal-genislik", "petal-uzunluk", "petal-genislik","sinif"]
    dataset = read_csv(url, names=isim)
    print("cvs dosyasi okuma basarili.")
except Exception as e:
    print("cvs dosyasi okunamadı")