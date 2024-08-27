#kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#kodlar
#veri yukleme

veriler = pd.read_csv('eksikveriler.csv')
#pd.read_csv("veriler.csv")

print(veriler)

#veri on isleme

boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)

x = 10

class insan:
    boy = 180
    def kosmak(self,b):
        return b + 10

ali = insan()
print(ali.boy)
print(ali.kosmak(90))

#eksik veriler
#sci - kit learn

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

"""
LabelEncoder, scikit-learn kütüphanesinde bulunan ve kategorik verileri 
sayısal verilere dönüştürmek için kullanılan bir araçtır. 
Makine öğrenmesi algoritmaları genellikle sayısal verilerle çalıştığı için, 
kategorik verilerin sayısal verilere dönüştürülmesi gerekmektedir. 
LabelEncoder, bu dönüşümü kolay ve etkili bir şekilde yapar.

LabelEncoder'ın başlıca işlevleri şunlardır:

Kategorik Verilerin Kodlanması: Kategorik verileri, her kategoriye bir tamsayı değeri 
atayarak sayısal değerlere dönüştürür. Örneğin, ['kırmızı', 'mavi', 'yeşil'] kategorilerini
 [0, 1, 2] gibi sayısal değerlere çevirir.

Model Uyumlaştırma: Sayısal verilere dönüştürülen kategorik veriler, 
makine öğrenmesi modelleri tarafından daha kolay anlaşılır ve işlenir hale gelir.

Veri Ön İşleme: Veri ön işleme adımlarının bir parçası olarak, 
etiketlerin sayısal verilere dönüştürülmesi, veriyi modeller için daha uygun hale getirir.
Örnek kullanım:

from sklearn.preprocessing import LabelEncoder

# Örnek kategorik veri
kategori_veri = ['kırmızı', 'mavi', 'yeşil', 'mavi', 'kırmızı']

# LabelEncoder'ı oluştur ve fit et
le = LabelEncoder()
sayısal_veri = le.fit_transform(kategori_veri)

print(sayısal_veri)
# Çıktı: array([0, 1, 2, 1, 0])

# Etiketleri geri dönüştürme
orijinal_veri = le.inverse_transform(sayısal_veri)
print(orijinal_veri)
# Çıktı: array(['kırmızı', 'mavi', 'yeşil', 'mavi', 'kırmızı'], dtype='<U6')

Bu örnekte, kırmızı, mavi ve yeşil kategorileri sırasıyla 0, 1 ve 2 sayısal değerlerine 
dönüştürülmüştür. Ayrıca, sayısal veriler tekrar orijinal kategorilere dönüştürülebilir.
Bu işlem, makine öğrenmesi projelerinde kategorik verilerin etkili bir şekilde 
kullanılmasını sağlar.
"""

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()

"""
OneHotEncoder, scikit-learn kütüphanesinde bulunan ve kategorik verileri 
sayısal verilere dönüştürmek için kullanılan bir diğer araçtır. 
Ancak, LabelEncoder'dan farklı olarak, OneHotEncoder

her kategori için ayrı bir sütun oluşturur ve bu sütunda sadece o kategoriye ait 
gözlemler için 1, diğerleri için ise 0 değeri yer alır. 
Bu yöntem, kategorik değişkenlerin modelde yanlış anlaşılmasını engeller ve birçok makine 
öğrenmesi algoritması için daha uygun hale getirir.

OneHotEncoder'ın Başlıca İşlevleri:
Kategorik Verilerin İkili (Binary) Vektörlere Dönüştürülmesi: Her kategoriyi ayrı bir sütun
olarak temsil eder ve bu sütunda o kategori için 1, diğer kategoriler için 0 değeri 
kullanılır.

Model Uyumlaştırma: Sayısal verilere dönüştürülen kategorik veriler, 
özellikle lineer modeller ve sinir ağları gibi algoritmalar için daha anlamlı hale gelir.

Veri Ön İşleme: Kategorik verilerin ikili vektörlere dönüştürülmesi, 
veri setinin homojen hale gelmesini sağlar ve modellerin bu verilerle daha etkili 
çalışmasına olanak tanır.

from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Örnek kategorik veri
kategori_veri = np.array(['kırmızı', 'mavi', 'yeşil', 'mavi', 'kırmızı']).reshape(-1, 1)

# OneHotEncoder'ı oluştur ve fit et
ohe = OneHotEncoder()
ohe_veri = ohe.fit_transform(kategori_veri).toarray()

print(ohe_veri)

[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]]


 Bu örnekte, kırmızı, mavi ve yeşil kategorileri sırasıyla üç sütunlu bir vektör olarak temsil edilir. Her satır, orijinal kategorik değere karşılık gelen bir ikili vektördür.

Farklı Kategoriler İçin Sütunlar:
kırmızı -> [1. 0. 0.]
mavi -> [0. 1. 0.]
yeşil -> [0. 0. 1.]

OneHotEncoder, bu şekilde kategorik verilerin her birini ayrı bir özellik (feature) olarak
temsil eder ve bu özelliklerin birbirinden bağımsız olduğunu belirtir. 
Bu yöntem, özellikle kategorik veriler arasında sıralı bir ilişki olmadığı 
durumlarda kullanışlıdır.

"""


ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

print(list(range(22)))
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)

































