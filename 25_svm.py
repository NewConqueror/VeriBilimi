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
svc = SVC(kernel='poly')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)

"""

SVC (Support Vector Classifier), destek vektör makineleri (SVM - Support Vector Machine) 
algoritmasının bir sınıflandırma probleminde kullanılan versiyonudur. 
SVM, denetimli öğrenme yöntemlerinden biri olup, sınıflandırma ve regresyon problemlerinde
kullanılan güçlü ve esnek bir makine öğrenmesi algoritmasıdır.

SVC Nasıl Çalışır?
SVC'nin temel amacı, iki veya daha fazla sınıf arasında en iyi ayırıcı hattı (hiper düzlem) 
bulmaktır. Bu hiper düzlem, farklı sınıflardaki verileri en iyi şekilde ayırır. 
SVC, bu ayırıcı hattı bulurken, sınıflar arasındaki en yakın veri noktalarına olan 
mesafeyi maksimum yapmaya çalışır. Bu en yakın veri noktalarına "destek vektörleri" denir.

Temel Kavramlar:
Hiper Düzlem (Hyperplane): Veriyi sınıflandırmak için kullanılan karar sınırıdır. 
İki boyutlu bir uzayda bu bir doğru, üç boyutlu bir uzayda ise bir düzlem olur. 
Daha yüksek boyutlu uzaylarda ise hiper düzlem olarak adlandırılır.

Marjin (Margin): SVC'nin en önemli özelliklerinden biri, sınıflar arasında maksimum marjin 
(en yakın veri noktalarına olan mesafe) sağlamaya çalışmasıdır. Marjin ne kadar büyük olursa, 
model o kadar iyi genelleme yapar.

Destek Vektörleri (Support Vectors): Marjinin sınırında bulunan veri noktalarına verilen 
isimdir. Bu noktalar, hiper düzlemin pozisyonunu belirler. Diğer noktalar hiper düzlemin 
yerini etkilemez.

Kernel Fonksiyonu: Verilerin doğrusal olarak ayrılmadığı durumlarda, SVM doğrusal olmayan 
bir ayrım elde etmek için "kernel trick" kullanır. Kernel fonksiyonu, 
veriyi daha yüksek boyutlu bir uzaya dönüştürerek doğrusal bir ayrım yapmayı sağlar. 
kernel fonksiyonları arasında lineer kernel, polinomial kernel, RBF (radial basis function) 
kernel, ve sigmoid kernel bulunur.

SVC'nin Avantajları:

Genel Performans: Yüksek doğruluğa sahip ve özellikle yüksek boyutlu veri setlerinde 
iyi çalışır.
Verimli: Verilerin sadece bir alt kümesi (destek vektörleri) kullanılarak eğitildiği için 
hesaplama açısından verimlidir.
Esneklik: Kernel fonksiyonları sayesinde doğrusal olmayan ayrımları da modelleyebilir.

SVC'nin Dezavantajları:
Hesaplama Maliyeti: Büyük veri setlerinde ve çok sayıda özellik içeren veri setlerinde 
eğitimi yavaş olabilir.
Hiperparametre Ayarı: Kernel tipi, C (regülarizasyon) ve gamma gibi hiperparametrelerin 
dikkatli bir şekilde ayarlanması gerekir. Yanlış seçimler model performansını olumsuz 
etkileyebilir.
Yorumlanabilirlik: Model, özellikle karmaşık kernel fonksiyonları kullanıldığında, 
karar mekanizmasını açıklamak zor olabilir.

Uygulama Alanları:
SVC, yüz tanıma, metin sınıflandırma, biyoinformatik, görüntü işleme gibi birçok alanda 
yaygın olarak kullanılır. Özellikle verilerin yüksek boyutlu ve doğrusal olmayan bir yapıda 
olduğu durumlarda etkili bir sınıflandırma aracı olarak öne çıkar.

"""









