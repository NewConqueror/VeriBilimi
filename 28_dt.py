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

"""
Decision Tree (Karar Ağacı), makine öğrenmesinde ve veri madenciliğinde kullanılan 
bir modeldir. Hem sınıflandırma hem de regresyon problemlerinde uygulanabilir. 
Bu model, veriyi karar düğümleri ve dallar aracılığıyla ayrıştırarak nihai bir karara ulaşır.

Decision Tree'nin Temel Özellikleri:
Kök Düğüm (Root Node): Ağacın en üstündeki düğümdür. Bu düğüm, verinin ilk olarak ayrıldığı 
noktayı temsil eder. Genellikle, en bilgilendirici özellik burada kullanılır.

İç Düğüm (Internal Nodes): Kök düğümden sonra gelen ve daha fazla ayrımı temsil eden 
düğümlerdir. Her iç düğüm, bir özelliğe dayalı olarak veri setini 
daha küçük alt gruplara böler.

Yaprak Düğüm (Leaf Nodes): Karar ağacının en altındaki düğümlerdir ve nihai kararı veya 
sınıfı temsil eder. Yaprak düğümüne ulaşıldığında, veri setinin o dalı için tahmin yapılır.

Dallar (Branches): Dallar, düğümler arasındaki yolları temsil eder ve her dal, 
bir özelliğe dayalı olarak alınan kararı gösterir.

Nasıl Çalışır?

Özellik Seçimi: Ağacın her düğümünde, veri setini en iyi şekilde bölen özellik seçilir. 
Bu işlem genellikle Gini İmpurity, Entropy (bilgi kazancı) gibi ölçütlerle yapılır.

Bölme: Veri, seçilen özelliğe göre dallara ayrılır.

Tekrarlama: Bu işlem, yaprak düğümlerine ulaşılana kadar her dalda tekrar edilir. 
Her yeni düğümde, kalan veri alt kümesi üzerinde yeniden bölme yapılır.

Tahmin: Yeni bir veri noktası geldiğinde, bu nokta ağacın kök düğümünden başlayarak, 
yaprak düğümlerine kadar takip edilerek sınıflandırılır.

Avantajlar:
Kolay Yorumlanabilirlik: Karar ağaçları, insanların kolayca anlayabileceği 
ve yorumlayabileceği bir model sunar. Hangi özelliklerin nasıl kullanıldığını 
görselleştirmek mümkündür.
Az Veri Hazırlığı: Diğer algoritmalara kıyasla daha az veri ön işleme gerektirir. 
Kategorik veriler ve eksik verilerle iyi çalışır.
Hızlı ve Verimli: Küçük ve orta büyüklükteki veri setlerinde hızlıca sonuç verir.

Dezavantajlar:
Aşırı Uyum (Overfitting): Özellikle derin ağaçlar, eğitim verisine aşırı uyum sağlayabilir 
ve bu da modelin genelleme kabiliyetini düşürebilir.
Hassasiyet: Verideki küçük değişiklikler, ağacın yapısında büyük değişikliklere 
neden olabilir.
Karar ağaçları, makine öğrenmesi problemlerinde güçlü bir başlangıç noktasıdır 
ve özellikle daha karmaşık modellerin (örneğin, Random Forest) temelini oluşturur.
"""








