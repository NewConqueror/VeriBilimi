
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


"""
Bu kod, SVM modelini kullanarak sosyal ağ veriseti üzerinde bir sınıflandırma işlemi 
gerçekleştirir. Modelin performansı, hem confusion matrix ile değerlendirilir 
hem de k-katlamalı çapraz doğrulama ile daha sağlam bir doğrulama sağlanır. 
Çapraz doğrulama, modelin genellenebilirliğini ve farklı veri setlerinde nasıl performans 
göstereceğini anlamak için kullanılır.

"""

"""

K-katlamalı çapraz doğrulama (K-fold cross-validation), makine öğrenmesi modellerinin 
performansını değerlendirmek için yaygın olarak kullanılan bir yöntemdir. 
Bu yöntem, modelin bir veri seti üzerinde aşırı öğrenme (overfitting) yapıp yapmadığını 
kontrol etmek ve modelin genelleme yeteneğini (yani yeni, görülmemiş veriler üzerinde nasıl
performans göstereceğini) daha doğru bir şekilde değerlendirmek için kullanılır.

K-Katlamalı Çapraz Doğrulamanın İşleyişi:
Verinin Bölünmesi:

Veri seti, K sayıda alt kümeye (fold) bölünür. Örneğin, eğer K=5 ise, 
veri seti 5 eşit parçaya bölünür.

Model Eğitimi ve Testi:

Model, her seferinde bir alt kümeyi test seti olarak kullanırken kalan K-1 alt küme üzerinde 
eğitilir. Bu işlem K kez tekrarlanır, her seferinde farklı bir alt küme test seti 
olarak seçilir.

Sonuçların Birleştirilmesi:

Her bir iterasyondan elde edilen model performansı (örneğin, doğruluk, hassasiyet, 
F1 skoru gibi) kaydedilir.
Son olarak, tüm K iterasyonundaki performans metriklerinin ortalaması ve varyansı hesaplanır.

Neden K-Katlamalı Çapraz Doğrulama Kullanılır?

Daha Doğru Performans Tahmini: K-katlamalı çapraz doğrulama, modelin tüm veri seti üzerinde 
eğitilmeden ve test edilmeden önce, farklı alt kümelerdeki performansını değerlendirerek 
daha güvenilir bir performans tahmini sağlar.

Overfitting'in Önlenmesi: Bu yöntem, modeli aynı veri seti üzerinde tekrar tekrar eğitip 
test etmekten kaçınarak, overfitting riskini azaltır.

Verimlilik: K-katlamalı çapraz doğrulama, mevcut veri setinin en verimli şekilde 
kullanılmasını sağlar, çünkü her veri parçası hem eğitim hem de test seti olarak kullanılır.

Örnek:
Eğer bir veri setinde 1000 örnek varsa ve K=5 seçildiyse:

Veri seti, her biri 200 örnek içeren 5 alt kümeye bölünür.
İlk iterasyonda, model 800 örnek üzerinde eğitilir ve kalan 200 örnek üzerinde test edilir.
İkinci iterasyonda, başka bir 200'lük grup test seti olarak seçilir ve kalan 800 örnek 
üzerinde model eğitilir.
Bu işlem 5 kez tekrarlanır ve sonunda 5 ayrı performans metriği elde edilir.
Bu 5 sonuç ortalaması, modelin genel performansını daha doğru bir şekilde yansıtır.
Özet:
K-katlamalı çapraz doğrulama, veri setinin farklı alt kümelerinde model performansını 
ölçerek, modelin genellenebilirliğini değerlendiren güçlü bir tekniktir. 
Modelin her bir alt küme üzerinde test edilmesi, daha sağlam bir performans tahmini sağlar 
ve aşırı öğrenme riskini azaltır.

"""




