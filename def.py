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
print("_____________________________")
# DATASETI URL'DEN CEK VE CVS HALINDE OKU
try:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    isim = ["sepal-uzunluk", "sepal-genislik", "petal-uzunluk", "petal-genislik","sinif"]
    dataset = read_csv(url, names=isim)
    print("cvs dosyasi okuma basarili.")
    print("_____________________________")
except Exception as e:
    print("cvs dosyasi okunamadı")
    print("_____________________________")

#DATASETIN Boyutunu
print("Dataset'in boyutu :" + str(dataset.shape))
print("_____________________________")

##DATASETIN 20 NESNESINE BAK
#print(dataset.head(20))

##DATASETIN ISTATIKSEL OZETI
print(dataset.describe())
print("_____________________________")

## SINIF DAGILIMI
print(dataset.groupby('sinif').size())
print("_____________________________")

##DATA GORSELLEME KUTU GRAFİĞİ
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False,sharey=False)

#HISTOGRAM GRAFİĞİ
#dataset.hist()
#pyplot.show()

#SCATTER MATRIX GRAFIGI
#scatter_matrix(dataset)
#pyplot.show()

##DATASETI %80 TRAİN %20 TEST AYIRIYORUZ.
array = dataset.values
X = array[:,0:4] # 4CU ROW HARIC BUTUN INTLERI AL
Y = array[:,4]  # SADECE 4CU ROWUN HEPSINI AL
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20,random_state=1)


#6 DEĞİŞİK ALGOYU ACCURACY SKORUNA GORE YAZDIR
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
sonuclar = []
isimler = []
for isim,model in models:
    kfold = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
    cv_results = cross_val_score(model,X_train, Y_train, cv=kfold,scoring='accuracy')
    sonuclar.append(cv_results)
    isimler.append(isim)
    print("%s: %f (%f)" % (isim, cv_results.mean(), cv_results.std()))

##ALGORITMA SONUCLARI KIYAASLA
#pyplot.boxplot(sonuclar, labels=isimler)
#pyplot.title('Algoritma kiyaslamasi')
#pyplot.show()


#TAHMIN URET
model = SVC(gamma='auto')
model.fit(X_train,Y_train)
tahmin = model.predict(X_test)
print(accuracy_score(Y_test, tahmin))
print(confusion_matrix(Y_test, tahmin))
print(classification_report(Y_test, tahmin))
