#kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri kümesi
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# eğitim ve test kümelerinin bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Tahminler
y_pred = classifier.predict(X_test)

#  Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)


#k-katlamali capraz dogrulama 
from sklearn.model_selection import cross_val_score
''' 
1. estimator : classifier (bizim durum)
2. X
3. Y
4. cv : kaç katlamalı

'''
basari = cross_val_score(estimator = classifier, X=X_train, y=y_train , cv = 4)
print(basari.mean())
print(basari.std())



# parametremetre optimizasyonu ve algoritma seçimi
from sklearn.model_selection import GridSearchCV
p = [{'C':[1,2,3,4,5],'kernel':['linear']},
     {'C':[1,2,3,4,5] ,'kernel':['rbf'],
      'gamma':[1,0.5,0.1,0.01,0.001]} ]

'''
GSCV parametreleri
estimator : sınıflandırma algoritması (neyi optimize etmek istediğimiz)
param_grid : parametreler/ denenecekler
scoring: neye göre skorlanacak : örn : accuracy
cv : kaç katlamalı olacağı
n_jobs : aynı anda çalışacak iş
'''
gs = GridSearchCV(estimator= classifier, #SVM algoritması
                  param_grid = p,
                  scoring =  'accuracy',
                  cv = 10,
                  n_jobs = -1)

grid_search = gs.fit(X_train,y_train)
eniyisonuc = grid_search.best_score_
eniyiparametreler = grid_search.best_params_

print(eniyisonuc)
print(eniyiparametreler)

"""
Izgara Araması (Grid Search), makine öğrenmesinde 'hiperparametre optimizasyonu' için 
kullanılan bir yöntemdir. Modelin performansını artırmak için, belirli hiperparametrelerin 
en iyi kombinasyonlarını bulmak amacıyla sistematik bir arama yapar. Bu yöntem, 
modelin genel performansını optimize etmek için hiperparametrelerin çeşitli değerlerini 
denemek için kullanılır.

Grid Search'in İşleyişi:
Hiperparametrelerin Belirlenmesi:

Modelin hiperparametrelerini belirler ve her bir hiperparametre için değerler 
(veya değer aralıkları) tanımlar. Örneğin, bir destek vektör makinesi (SVM) için C ve gamma
hiperparametrelerinin değerleri belirlenir.

Parametre Kombinasyonlarının Oluşturulması:

Tüm hiperparametrelerin olası değerlerinin kombinasyonları oluşturulur. 
Bu, bir "ızgara" gibi düşünülür, burada her hücre bir hiperparametre kombinasyonunu 
temsil eder.
Modelin Eğitilmesi ve Test Edilmesi:

Her bir hiperparametre kombinasyonu için model eğitilir ve test edilir. 
Bu genellikle k-katlamalı çapraz doğrulama (k-fold cross-validation) kullanılarak yapılır.

Performansın Değerlendirilmesi:
Her bir hiperparametre kombinasyonu için modelin performansı değerlendirilir. 
Bu genellikle doğruluk, F1 skoru, hassasiyet gibi performans metrikleri ile yapılır.

En İyi Kombinasyonun Seçilmesi:
En iyi performansı gösteren hiperparametre kombinasyonu seçilir. Bu kombinasyon, 
modelin en iyi şekilde performans göstermesi beklenen hiperparametrelerdir.

Grid Search'in Avantajları:
Kapsamlı Arama: Belirlenen hiperparametrelerin tüm kombinasyonlarını denediği için geniş 
bir arama yapabilir.
Sistematik: Her bir kombinasyon belirli bir düzen içinde test edilir, 
bu da daha güvenilir sonuçlar sağlar.

Grid Search'in Dezavantajları:
Zaman ve Hesaplama Maliyeti: Özellikle büyük veri setleri ve çok sayıda hiperparametre ile 
çalışırken hesaplama süresi ve kaynak kullanımı yüksek olabilir.
Parametre Uzayının Büyüklüğü: Hiperparametrelerin sayısı ve her birinin alabileceği değerler 
arttıkça, kombinasyonların sayısı üssel olarak artar. Bu, hesaplama süresini önemli ölçüde 
artırabilir.


Bir modelin C ve gamma hiperparametrelerini optimize etmek istediğinizi varsayalım:

C için değerler: [0.1, 1, 10]
gamma için değerler: [0.001, 0.01, 0.1]
Grid Search, toplamda 3 x 3 = 9 farklı kombinasyonu deneyecek ve her kombinasyon için 
modelin performansını değerlendirecektir.

"""









