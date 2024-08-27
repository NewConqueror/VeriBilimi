#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)



from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)

"""
Random Forest Classifier, makine öğrenmesinde yaygın olarak kullanılan güçlü bir topluluk 
(ensemble) yöntemidir. Bu yöntem, sınıflandırma problemlerinde kullanılır ve birçok karar 
ağacının bir araya getirilmesiyle çalışır.

Random Forest'in Temel Özellikleri:
Topluluk Yöntemi: Birden fazla karar ağacından (decision tree) oluşur. Her bir karar ağacı, 
veri setinin farklı bir alt kümesi üzerinde eğitilir.

Çeşitlendirme (Bagging): Her ağaç, rastgele bir veri alt kümesi (bootstrapping) ve özellik 
alt kümesi kullanılarak eğitilir. Bu, modellerin birbirinden bağımsız olmasını ve 
aşırı uyum (overfitting) riskini azaltır.

Çoğunluk Oyu: Sınıflandırma işlemi sırasında, her bir karar ağacı bir tahminde bulunur 
ve en çok oy alan sınıf, nihai tahmin olarak seçilir.

Dayanıklılık: Aşırı uyum ve gürültüye karşı dayanıklıdır, bu da onu çeşitli veri setlerinde 
iyi performans gösteren bir model yapar.

Avantajlar:
Aşırı Uyum (Overfitting) Azalır: Birden fazla ağacın kullanılması, modelin genelleme 
kabiliyetini artırır.
Özellik Seçimi: Özelliklerin önemli olup olmadığını belirlemeye yardımcı olur.
Hızlı ve Esnek: Hem büyük veri setlerinde hem de çok boyutlu özellik uzaylarında iyi çalışır.

Dezavantajlar:
Yorumlanabilirlik: Karar ağaçları tek başına basit ve yorumlanabilir olmasına rağmen, 
birden fazla ağacın bir araya gelmesiyle oluşan model, karmaşık ve zor yorumlanabilir
hale gelir.
Hafıza ve İşlem Gücü Gereksinimi: Çok sayıda ağaç eğitmek ve tahmin yapmak için daha fazla 
hesaplama gücü ve hafıza gerekir.
Random Forest, sınıflandırma dışında regresyon problemlerinde de kullanılabilir 
(Random Forest Regressor olarak bilinir). Makine öğrenmesinde, özellikle veri setlerinde 
çok fazla değişkenlik olduğunda veya modelin genelleme kabiliyetine ihtiyaç duyulduğunda, 
yaygın olarak tercih edilir.

"""







