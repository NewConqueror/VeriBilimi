from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

"""
Bu kod, yüz tanıma ve konut fiyat tahmini gibi iki farklı probleme yönelik olarak 
Rastgele Orman (Random Forest) algoritmasını uygulamaktadır. 
Kod iki ana bölümden oluşuyor:

Olivetti Yüz Veri Seti: fetch_olivetti_faces fonksiyonu, yüz tanıma için kullanılan 
Olivetti veri setini yükler. Bu veri seti, 64x64 piksel boyutunda gri tonlamalı 400 yüz 
görüntüsü içerir, her biri farklı kişilerden alınmıştır.

Görselleştirme: İlk iki yüz görüntüsü, imshow kullanılarak görselleştirilir ve görüntüler 
gray (gri tonlama) renk haritasıyla gösterilir.


Veri Dönüştürme: Olivetti veri setindeki her bir 64x64 piksel boyutundaki görüntü, 
1D (4096 boyutlu) bir vektöre dönüştürülür (X = oli.data). Bu, görüntüdeki her pikselin 
bir özellik (feature) olarak kabul edildiği anlamına gelir.

Eğitim ve Test Setlerinin Ayrılması: Veri seti, train_test_split ile %80 eğitim, 
%20 test olarak bölünür (test_size=0.2).
Rastgele Orman Sınıflandırıcısı: RandomForestClassifier sınıfı ile bir 
Rastgele Orman modeli oluşturulur. Model, 100 karar ağacından (n_estimators=100) oluşur 
ve eğitim verisiyle (X_train, y_train) eğitilir.

"""

oli = fetch_olivetti_faces()

"""
    2D (64X64) -> 1D (4096)
"""

plt.figure()
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(oli.images[i], cmap = "gray")
    plt.axis("off")
plt.show()

X = oli.data 
y = oli.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

rf_clf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Acc: ", accuracy)

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


"""
California Konut Veri Seti: fetch_california_housing fonksiyonu, 
California'daki konut fiyatlarının tahmin edilmesi için kullanılan veri setini yükler. 
Bu veri seti, çeşitli coğrafi ve demografik özelliklere (örneğin, nüfus, hane sayısı) 
dayalı olarak konut fiyatlarının tahmin edilmesi amacıyla kullanılır.
"""

california_housing = fetch_california_housing()

X = california_housing.data
y = california_housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
rf_reg = RandomForestRegressor(random_state = 42)
rf_reg.fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("rmse: ", rmse)

"""

Hata Hesaplama: Tahmin edilen fiyatlar ile gerçek fiyatlar arasındaki 
ortalama karesel hata (mean_squared_error, MSE) hesaplanır.

Kök Ortalama Karesel Hata (RMSE): MSE'nin karekökü alınarak kök ortalama karesel hata (RMSE) 
elde edilir. RMSE, modelin tahminlerinin ne kadar doğru olduğunu gösterir ve sonuç ekrana 
yazdırılır.

Bu kod, yüz tanıma ve konut fiyat tahmini problemlerine yönelik olarak 
Rastgele Orman algoritmasının nasıl uygulanacağını ve modelin doğruluğunu veya hatasını 
nasıl değerlendireceğinizi göstermektedir. İlk kısımda, yüz tanıma için bir sınıflandırma 
modeli oluşturulurken, ikinci kısımda konut fiyatlarını tahmin etmek için bir regresyon 
modeli kullanılmaktadır.

"""