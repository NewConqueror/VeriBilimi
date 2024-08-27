#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri kümesi
veriler = pd.read_csv('Wine.csv')
X = veriler.iloc[:, 0:13].values
y = veriler.iloc[:, 13].values

# eğitim ve test kümelerinin bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

#pca dönüşümünden önce gelen LR
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#pca dönüşümünden sonra gelen LR
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

#tahminler
y_pred = classifier.predict(X_test)

y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
#actual / PCA olmadan çıkan sonuç
print('gercek / PCAsiz')
cm = confusion_matrix(y_test,y_pred)
print(cm)

#actual / PCA sonrası çıkan sonuç
print("gercek / pca ile")
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)

#PCA sonrası / PCA öncesi
print('pcasiz ve pcali')
cm3 = confusion_matrix(y_pred,y_pred2)
print(cm3)

"""
Bu kod, bir şarap veri kümesi üzerinde Principal Component Analysis (PCA) adlı bir boyut 
indirgeme yöntemi uygulayarak, lojistik regresyon modeli ile sınıflandırma yapmayı ve 
PCA kullanımı öncesi ve sonrası elde edilen sonuçları karşılaştırmayı amaçlar.

Bu kod, PCA ile veri boyutunu indirgedikten sonra lojistik regresyon modelinin 
performansını değerlendirir ve PCA kullanımı öncesi ve sonrası tahminlerin doğruluğunu 
karşılaştırır. Bu, modelin daha az özellik kullanarak (yani, boyut indirgemesi sonrası) 
ne kadar iyi performans gösterdiğini anlamak için yararlıdır.
"""

"""

PCA (Principal Component Analysis), yani Temel Bileşen Analizi, 
bir veri setindeki yüksek boyutlu verileri daha düşük boyutlu bir temsil haline getirmek 
için kullanılan yaygın bir boyut indirgeme tekniğidir. PCA, orijinal veri setindeki 
en önemli bilgiyi koruyarak verilerin boyutunu azaltır. Bu, veri analizini, 
görselleştirmeyi ve modelleme işlemlerini kolaylaştırır.

PCA'nın Temel Prensipleri:

Veri Setinin Standartlaştırılması:

PCA uygulamadan önce, veri setindeki her bir özelliğin (değişkenin) ortalamasını 0 yapacak 
ve standart sapmasını 1 yapacak şekilde standartlaştırılması (ölçeklenmesi) gerekir. 
Bu, tüm değişkenlerin eşit ağırlıkta işlem görmesini sağlar.

Kovaryans Matrisinin Hesaplanması:
Standartlaştırılmış veri seti üzerinde, özelliklerin birbiriyle nasıl ilişkili olduğunu 
gösteren bir kovaryans matrisi oluşturulur. Kovaryans matrisi, 
verinin farklı özelliklerinin birbirleriyle olan ilişkisinin yönünü ve gücünü gösterir.

Özdeğerler ve Özvektörlerin Hesaplanması:
Kovaryans matrisinden özdeğerler ve özvektörler hesaplanır. Özvektörler, 
verinin yönünü gösterirken, özdeğerler bu yönlerdeki varyansın büyüklüğünü temsil eder.
Özvektörler, temel bileşenler olarak adlandırılır ve her bir temel bileşen verinin 
orijinal uzayda ne kadar varyans taşıdığını ifade eder.

Temel Bileşenlerin Seçilmesi:
Hesaplanan özvektörler arasından en büyük özdeğere sahip olanlar seçilir. 
Bu temel bileşenler, verinin maksimum varyansını koruyarak orijinal 
veri uzayındaki boyutu azaltır.
Genellikle, toplam varyansın büyük bir yüzdesini (örneğin, %95) açıklayan bileşenler seçilir.


Verinin Yeni Bileşen Uzayına Dönüştürülmesi:
Orijinal veri, seçilen temel bileşenlere projekte edilir. 
Bu projeksiyon, veriyi düşük boyutlu bir uzayda temsil eder.

PCA'nın Uygulamaları:
Boyut İndirgeme:
PCA, büyük boyutlu veri setlerinde (çok sayıda özellik içeren) boyut indirgeme yaparak 
verilerin daha basit bir şekilde analiz edilmesini sağlar.

Özellikle çok sayıda özellik olduğunda, PCA kullanarak 
daha az sayıda bileşen ile veri temsil edilebilir.

Görselleştirme:
Yüksek boyutlu verilerin 2D veya 3D görselleştirilmesi için PCA kullanılır. 
Bu, özellikle kümelenme analizinde ve sınıflandırma problemlerinde 
verinin görselleştirilmesine yardımcı olur.

Gürültü Azaltma:

PCA, veri setindeki gürültüleri azaltarak, önemli bileşenleri seçer. 
Bu, modelleme süreçlerinde daha iyi sonuçlar elde edilmesini sağlar.

Özellik Seçimi:

PCA, veri setindeki en bilgilendirici özellikleri seçerek, modelin daha verimli çalışmasını
sağlar. Bu, özellikle büyük veri setlerinde modelin karmaşıklığını azaltır.

Özet:
PCA, veri setindeki en önemli bilgileri koruyarak verilerin boyutunu azaltan güçlü bir 
istatistiksel yöntemdir. Bu yöntem, hem veri analizi hem de modelleme süreçlerinde sıkça 
kullanılır ve verilerin daha basit, daha anlaşılır bir temsilini sağlar. 
PCA, veri setindeki varyansı maksimize eden yeni değişkenler (temel bileşenler) oluşturur 
ve bu yeni değişkenler, orijinal verilerin doğrusal kombinasyonlarıdır.

"""










