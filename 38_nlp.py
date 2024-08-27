import numpy as np
import pandas as pd


# Verilerin Yüklenmesi
yorumlar = pd.read_csv('Restaurant_Reviews.csv')

"""Kütüphanelerin Yüklenmesi ve Ön Hazırlık:

nltk kütüphanesi, metin işleme için kullanılıyor.
PorterStemmer, kelimeleri köklerine indirmek (stemming) için kullanılıyor.
stopwords, İngilizce dilindeki yaygın olarak kullanılan ve anlam taşımayan 
kelimeleri filtrelemek için kullanılıyor (örneğin "and", "the").
"""
import re
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

#Preprocessing (Önişleme)
derlem = []
for i in range(1000):
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)
    
#Feautre Extraction ( Öznitelik Çıkarımı)
#Bag of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(derlem).toarray() # bağımsız değişken
y = yorumlar.iloc[:,1].values # bağımlı değişken
 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

"""
Özet: Bu kod, restoran yorumlarını alıp metin ön işleme teknikleriyle işliyor, 
ardından bir "Bag of Words" modeli oluşturuyor ve son olarak bu veriler üzerinde bir 
Naive Bayes modeli eğitip, test verisindeki yorumların olumlu mu yoksa olumsuz mu olduğunu 
tahmin ediyor. Modelin başarısı ise bir karışıklık matrisi ile değerlendiriliyor.
"""

"""
Doğal Dil İşleme (Natural Language Processing - NLP), bilgisayarların insan dilini anlama, 
yorumlama ve üretme yeteneğini geliştiren bir alan olarak tanımlanır. NLP, dilbilim, 
bilgisayar bilimi ve yapay zeka gibi disiplinlerin kesişiminde yer alır ve insan dilini 
bilgisayarlarla etkili bir şekilde işlemek için gerekli teknik ve algoritmaları geliştirir.

NLP'nin Temel Amaçları:
Dil Anlama (Language Understanding): Bilgisayarların insan dilini anlamasını sağlamak. 
Bu, metinlerin anlamını çıkarma, duygu analizi, anlamsal benzerlikler bulma gibi işlemleri 
içerir.

Dil Üretme (Language Generation): Bilgisayarların insan dilinde anlamlı metinler üretmesini 
sağlamak. Örneğin, metin oluşturma, özetleme veya diyalog sistemleri bu amaca hizmet eder.

NLP'nin Temel Bileşenleri:
Metin Ön İşleme: Metni analiz etmek için uygun formata getirme sürecidir. 
Bu, metin temizleme, kelime köklerine ayırma (stemming/lemmatization), 
stopword'leri çıkarma gibi işlemleri içerir.

Dil Modelleri: Doğal dildeki kalıpları ve kuralları öğrenen modellerdir. 
Örneğin, kelime vektörleri (Word2Vec, GloVe), dil modelleri (BERT, GPT) gibi.

Öznitelik Çıkarımı: Metni sayısal verilere dönüştürme işlemidir. Bu işlem, 
kelime frekansları (Bag of Words), TF-IDF (Term Frequency-Inverse Document Frequency) gibi 
tekniklerle yapılır.

Sınıflandırma ve Tahmin: Metinlerin sınıflandırılması veya belirli özelliklerin tahmin 
edilmesi işlemleridir. Bu, metinlerin olumlu/olumsuz olarak sınıflandırılması veya 
metin bazlı tahminler yapılması anlamına gelir.

Doğal Dil Anlama (NLU - Natural Language Understanding): Metnin anlamını çıkarma ve sorulara 
yanıt verme gibi daha karmaşık işlemler içerir.

Doğal Dil Üretimi (NLG - Natural Language Generation): İnsan dilinde anlamlı cümleler veya 
paragraflar üretme işlemidir. Örneğin, otomatik metin oluşturma, rapor yazma gibi.

NLP Uygulama Alanları:
Duygu Analizi: Sosyal medya gönderileri veya müşteri yorumları gibi verilerdeki duygusal 
eğilimleri tespit etmek.
Makine Çevirisi: Bir dildeki metni otomatik olarak başka bir dile çevirmek 
(örneğin, Google Translate).
Sohbet Robotları ve Sanal Asistanlar: Kullanıcılarla doğal dilde etkileşime giren sistemler 
(örneğin, Siri, Alexa).
Otomatik Özetleme: Uzun metinleri özetleyen sistemler.
Adli Dilbilim: Metin tabanlı suç araştırmaları ve kimlik tespitinde kullanılır.
Sonuç:
NLP, bilgisayarların insan dilini anlamasını ve işlemesini sağlayarak, insan-bilgisayar 
etkileşimini daha doğal ve etkili hale getirir. Bu alanda yapılan ilerlemeler, 
birçok günlük uygulamanın temelini oluşturur ve sürekli gelişmektedir.

"""

















