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



"""

Naive Bayes Algoritması Nasıl Çalışır?
Naive Bayes sınıflandırıcısı, verilen bir veri noktası için hangi sınıfa ait olduğuna karar 
vermek için yukarıdaki Bayes teoremini kullanır. Algoritmanın işleyişi şu şekildedir:

Öncelik (Prior) Olasılıklarının Hesaplanması: Eğitim verisindeki her sınıfın frekansına 
dayalı olarak, her bir sınıf için öncelik olasılıkları hesaplanır.

Olasılıkların Hesaplanması: Her bir sınıf için, verilen özelliklerin sınıfa ait olma 
olasılıkları hesaplanır. Bu işlem, her bir özelliğin sınıfa bağımsız olarak katkıda 
bulunduğu varsayımına dayanır.

Posterior Olasılıkların Hesaplanması: Bayes teoremi kullanılarak, her sınıf için posterior 
olasılıklar hesaplanır.

Sınıflandırma: Hesaplanan posterior olasılıklar arasından en yüksek olasılığa sahip sınıf 
seçilerek, veri noktası bu sınıfa atanır.

Naive Bayes Türleri:
Naive Bayes algoritmasının çeşitli türleri vardır. Bu türler, özelliklerin türüne ve 
dağılımına göre farklılık gösterir:

Gaussian Naive Bayes: Sürekli özelliklerin normal dağıldığı varsayılır. 
Özellikle özelliklerin sürekli olduğu ve normal dağılıma uyduğu durumlarda kullanılır.

Multinomial Naive Bayes: Özellikle metin sınıflandırması gibi ayrık özelliklerin bulunduğu 
durumlarda kullanılır. Burada, özellikler sınıfların olasılıkları ile ilişkili kelime sayısı 
gibi düşünülür.

Bernoulli Naive Bayes: Özelliklerin ikili (0 veya 1) olduğu durumlarda kullanılır. 
Bu model, örneğin bir metinde belirli bir kelimenin var olup olmadığını kontrol eden metin 
sınıflandırması görevlerinde kullanılır.

Avantajlar:
Hızlı ve Verimli: Özellikle büyük veri setlerinde hızlı ve etkili çalışır.
Az Veri Gereksinimi: Diğer algoritmalara kıyasla daha az veriyle iyi sonuçlar 
elde edilebilir.
Kullanımı Kolay: Uygulaması ve eğitimi oldukça basittir.

Dezavantajlar:
Bağımsızlık Varsayımı: Özellikler arasındaki bağımsızlık varsayımı gerçekte nadiren doğru 
olur, bu da modelin performansını sınırlayabilir.
Veri Dengesizliği: Sınıflar arasındaki büyük dengesizlikler, modelin doğruluğunu olumsuz 
etkileyebilir.
Uygulama Alanları:
Naive Bayes sınıflandırıcısı, özellikle metin sınıflandırma (örneğin spam tespiti), 
duygu analizi, belge kategorizasyonu ve tıbbi teşhis gibi birçok uygulama alanında yaygın 
olarak kullanılır. Basit yapısı ve hızlı çalışması sayesinde, genellikle ilk denenen 
algoritmalardan biridir.

"""