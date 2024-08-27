#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('Churn_Modelling.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

#veri on isleme

X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values



#encoder: Kategorik -> Numeric
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
                        )
X = ohe.fit_transform(X)
X = X[:,1:]




#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#3 Yapay Sinir ağı
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(6, init = 'uniform', activation = 'relu' , input_dim = 11))

classifier.add(Dense(6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss =  'binary_crossentropy' , metrics = ['accuracy'] )

classifier.fit(X_train, y_train, epochs=50)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print(cm)

"""
Bu kod, bir bankanın müşteri verilerini kullanarak, bir müşterinin bankadan ayrılıp 
ayrılmayacağını tahmin eden bir yapay sinir ağı modeli oluşturur. Model, eğitim verileriyle 
eğitildikten sonra test verileri üzerinde tahminler yapar ve bu tahminlerin doğruluğunu 
bir karışıklık matrisi kullanarak değerlendirir.
"""

"""
Yapay Sinir Ağı (YSA), biyolojik sinir ağlarının (örneğin, insan beyninin) çalışma 
prensiplerinden esinlenerek geliştirilen bir tür makine öğrenmesi modelidir. 
Yapay sinir ağları, büyük veri kümelerinden öğrenerek karmaşık problemleri çözebilir, 
sınıflandırma, regresyon, görüntü tanıma, dil işleme gibi çeşitli görevlerde kullanılabilir.

Yapay Sinir Ağlarının Temel Bileşenleri:
Nöronlar (Nodes veya Units):

Biyolojik sinir hücrelerinden esinlenilerek geliştirilmiş, bilgiyi işleyen temel birimlerdir.
Her nöron, bir veya daha fazla girdi alır, bu girdileri işler ve bir çıktı üretir.
Katmanlar (Layers):

Girdi Katmanı (Input Layer): Bu katman, modele verilerin girdiği katmandır. 
Her düğüm (nöron), bir özellik (öz nitelik) veya girdi veri noktasını temsil eder.
Gizli Katmanlar (Hidden Layers): Bu katmanlar, girdileri işleyen ve çıktı katmanına bilgi 
gönderen ara katmanlardır. Her gizli katman, girdi katmanından veya önceki gizli katmandan 
gelen bilgileri işler.
Çıktı Katmanı (Output Layer): Bu katman, modelin nihai tahminini veya çıktısını üretir. 
Bu katman, sınıflandırma problemi için sınıf etiketlerini veya regresyon için sürekli 
değerleri içerir.

Bağlantılar ve Ağırlıklar (Connections and Weights):

Nöronlar arasındaki bağlantılar,her bir girdi verisi için ağırlıklarla çarpılarak modellenir.
Ağırlıklar, nöronlar arasındaki sinyallerin ne kadar önemli olduğunu belirler. 
Ağırlıklar eğitim sırasında optimize edilir.

Aktivasyon Fonksiyonları (Activation Functions):

Aktivasyon fonksiyonları, bir nöronun toplam girdisini çıktı olarak dönüştürmek için 
kullanılır. Bu fonksiyonlar, modelin doğrusal olmayan ilişkileri öğrenmesini sağlar.
Yaygın aktivasyon fonksiyonları arasında ReLU (Rectified Linear Unit), 
sigmoid ve tanh yer alır.

Yapay Sinir Ağlarının Çalışma Prensibi:
İleri Yönlü Yayılma (Forward Propagation):

Girdi katmanından başlayarak, veriler her bir nöronun ağırlıklarıyla çarpılır, 
toplama işlemi yapılır ve aktivasyon fonksiyonu uygulanır. Bu işlem, her katmandaki nöronlar 
arasında gerçekleşir ve nihayetinde çıktı katmanına ulaşır.

Hata Hesaplama:

Çıktı katmanındaki tahmin edilen değer ile gerçek değer arasındaki fark (hata) hesaplanır. 
Bu hata, modelin ne kadar doğru olduğunu ölçer.

Geri Yayılım (Backpropagation):

Hesaplanan hatayı minimize etmek amacıyla, hata geriye doğru yayılır. Bu süreçte, 
her bir bağlantının ağırlığı, hatayı azaltacak şekilde güncellenir.
Optimizasyon algoritmaları (örneğin, gradient descent) bu ağırlıkları optimize etmek için 
kullanılır.

Modelin Eğitimi:

Model, ileri yönlü yayılma ve geri yayılım süreçleri tekrar edilerek eğitilir. 
Her bir iterasyon (epoch) modelin performansını iyileştirir.

YSA'nın Uygulama Alanları:
Görüntü Tanıma: Yüz tanıma, nesne tanıma gibi görevlerde kullanılır.
Doğal Dil İşleme (NLP): Metin sınıflandırma, dil çevirisi, duygu analizi 
gibi uygulamalarda yer alır.
Tahmin Modelleri: Finansal tahminler, hava durumu tahminleri gibi alanlarda kullanılır.
Otonom Sistemler: Otonom araçların navigasyonunda ve karar verme süreçlerinde kullanılır.
Özet:
Yapay Sinir Ağları, biyolojik sinir ağlarını model alarak geliştirilmiş, 
karmaşık veri yapıları ve ilişkileri öğrenebilen güçlü bir modeldir. Bu modeller, 
çok katmanlı yapıları sayesinde doğrusal olmayan karmaşık ilişkileri öğrenebilir ve 
geniş bir uygulama yelpazesinde kullanılır.









"""










