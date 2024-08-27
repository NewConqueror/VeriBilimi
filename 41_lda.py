# kutuphaneler
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


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components = 2)

X_train_lda = lda.fit_transform(X_train,y_train)
X_test_lda = lda.transform(X_test)

#LDA donusumunden sonra
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)

#LDA verisini tahmin et
y_pred_lda = classifier_lda.predict(X_test_lda)

#LDA sonrası / orijinal 
print('lda ve orijinal')
cm4 = confusion_matrix(y_pred,y_pred_lda)
print(cm4)

"""
Bu kod, PCA ve LDA gibi iki farklı boyut indirgeme yöntemini karşılaştırır ve her iki 
yöntemin de lojistik regresyon modeli ile kullanıldığında, 
modelin performansını nasıl etkilediğini değerlendirir. 
PCA, verinin varyansını maksimize ederken, LDA sınıflar arasındaki ayrımı maksimize eder. 
Bu nedenle, LDA genellikle sınıflandırma problemlerinde daha iyi sonuçlar verir. 
Kod, bu iki yöntemi uygulayarak, elde edilen sonuçları karşılaştırır.
"""


"""
LDA (Linear Discriminant Analysis), yani Doğrusal Ayrım Analizi, 
denetimli bir makine öğrenmesi algoritmasıdır ve sınıflandırma problemlerinde yaygın olarak 
kullanılır. LDA, sınıflar arasındaki ayrımı maksimize etmek için veri setindeki boyutları 
azaltan bir boyut indirgeme yöntemidir. PCA (Principal Component Analysis) ile benzer bir 
şekilde çalışsa da, LDA'nın amacı, veriyi sınıflandırmada daha etkili bir şekilde kullanmak 
için sınıflar arasındaki farkı artırmaktır.

LDA'nın Amacı:
LDA'nın temel amacı, farklı sınıfları en iyi şekilde ayırt eden bir alt uzay bulmaktır. 
Bu alt uzay, veri setindeki sınıflar arasındaki farkı en üst düzeye çıkarmak için 
tasarlanmıştır. Bu, LDA'nın özellikle sınıflandırma problemlerinde kullanılmasının 
ana nedenidir.

LDA'nın Temel Prensipleri:

Sınıf Ortalamalarının Hesaplanması:
Her sınıf için özelliklerin ortalamaları hesaplanır. Bu ortalamalar, 
sınıfın "merkez" noktalarını temsil eder.

Sınıf İçindeki Varyans (Scatter) ve Sınıflar Arası Varyans:

Sınıf içi varyans (Within-Class Scatter): Aynı sınıfa ait verilerin ne kadar yayıldığını 
(dağıldığını) ölçer.

Sınıflar arası varyans (Between-Class Scatter): Farklı sınıfların ortalamaları arasındaki 
uzaklıkları ölçer.

Optimum Ayrım Yüzeyinin Bulunması:
LDA, sınıflar arası varyansı maksimize ederken, sınıf içi varyansı minimize eden bir 
doğrusal kombinasyon bulur. Bu, verilerin daha ayrışmış ve iyi sınıflandırılmış olmasını 
sağlar.

Yeni Veri Temsili:
Orijinal veri seti, bulunan bu doğrusal kombinasyonlar (ayrım yüzeyleri) üzerinde yeniden 
ifade edilir. Bu sayede, veri seti daha az boyutlu bir uzayda temsil edilir ve sınıflar 
arasındaki ayrım güçlendirilmiş olur.

LDA'nın Uygulamaları:

Boyut İndirgeme:

LDA, yüksek boyutlu veri setlerini daha düşük boyutlu hale getirir, bu da 
veri görselleştirme ve işleme süresini azaltmak için önemlidir.

Sınıflandırma:

LDA, sınıflandırma algoritmaları (örneğin, lojistik regresyon, destek vektör makineleri) 
için özelliklerin boyutlarını azaltırken, sınıflar arasındaki ayrımı korur.

Gürültü Azaltma:

LDA, veri setindeki gürültüyü azaltarak daha iyi sınıflandırma performansı elde edilmesini 
sağlar.
LDA vs. PCA:
LDA: Denetimli bir yöntemdir ve sınıf etiketlerini kullanır. 
Sınıflar arasındaki ayrımı maksimize etmeye çalışır.
PCA: Denetimsiz bir yöntemdir ve yalnızca verinin genel varyansını maksimize eder. 
Sınıf etiketlerini dikkate almaz.

Özet:
LDA, sınıflandırma problemlerinde veri setinin boyutunu azaltarak, sınıflar arasındaki farkı 
maksimize eden bir yöntemdir. Özellikle sınıflandırma işlemlerinde etkili olan LDA, 
verilerin daha iyi ayrışmasını sağlayarak, sınıflandırma algoritmalarının performansını 
artırır.

"""










