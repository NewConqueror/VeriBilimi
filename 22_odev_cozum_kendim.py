#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score


"""
ne yapıcaz ünvan ünvan seviyesi kıdem puan dan maası tahmin edicez calisan ID gereksiz
bir anlamı yok
unvanı sadece label encoder kullanarak yapıcaz çünkü bir sıralama yapılabiliyor
unvan seviyesi kıdem puan zaten sayısal onlara dokunmicaz
maası tahmin etmeye çalışacak

genel olarak tek değişkenli ile daha iyi çalışıyorlar
Çalışmayanı P value ile elemen lazım ama ben onu unuttum şadi hoca öyle yaptı
hatam dursun diye düzeltmicem ödev çözüm e bak anlarsın zaten
ols ile R2 P value falan görebiliyorsun ama ben göremedim eklemeyi unuttuğum için

"""

# veri yukleme
veriler = pd.read_csv('maaslar_yeni.csv')

maaslar=veriler.iloc[:,-1]

from sklearn import preprocessing

yeniveri = veriler.iloc[:,2:5]


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(yeniveri,maaslar)

plt.scatter(yeniveri,maaslar, color="red")
plt.plot(yeniveri,lin_reg.predict(yeniveri), color = 'blue')
plt.show()

print('Linear R2 degeri')
print(r2_score(maaslar, lin_reg.predict(yeniveri)))


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
yeniveri_poly = poly_reg.fit_transform(yeniveri)
print(yeniveri)
lin_reg2 = LinearRegression()
lin_reg2.fit(yeniveri_poly,maaslar)
plt.scatter(yeniveri,maaslar,color = 'red')
plt.plot(yeniveri,lin_reg2.predict(poly_reg.fit_transform(yeniveri)), color = 'blue')
# plt.plot(yeniveri,lin_reg2.predict(yeniveri_poly), color = 'blue') aynı
plt.show()

print('Polynomial R2 degeri')
print(r2_score(maaslar, lin_reg2.predict(poly_reg.fit_transform(yeniveri))))
print(r2_score(maaslar, lin_reg2.predict(yeniveri_poly)))


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

yeniveri_olcekli = sc1.fit_transform(yeniveri)

sc2=StandardScaler()
maaslar_olcekli = np.ravel(sc2.fit_transform(maaslar.reshape(-1,1)))

"""
Python'da NumPy kütüphanesinin np.ravel fonksiyonu, bir dizi (array) girişini 
tek boyutlu bir diziye (flattened array) dönüştürmek için kullanılır. 
Başka bir deyişle, çok boyutlu bir diziyi, elemanlarının düz bir sıralı liste halinde 
yer aldığı tek bir boyuta indirger.

import numpy as np

# 2x3 boyutunda bir dizi oluşturalım
array = np.array([[1, 2, 3], [4, 5, 6]])

# Diziye np.ravel uygulayalım
flattened_array = np.ravel(array)

print(flattened_array)

Bu kod çalıştırıldığında, çıktı olarak düz bir dizi elde ederiz:

[1 2 3 4 5 6]

np.ravel ve np.flatten Arasındaki Farklar
np.ravel(), mümkünse orijinal veriyle bağlantılı bir görünüm (view) döner. 
Bu, bellekte yeni bir kopya oluşturmadığı anlamına gelir, sadece orijinal veri yapısını 
düz bir şekilde görüntüler.

np.flatten(), her zaman orijinal dizinin bir kopyasını döner. 
Bu, yeni bir düz dizi oluşturur ve orijinal veri ile bağlantılı değildir.
Bu fark, özellikle büyük veri setlerinde belleği verimli kullanmak açısından 
önemli olabilir.
"""


from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(yeniveri_olcekli,maaslar_olcekli)

plt.scatter(yeniveri_olcekli,maaslar_olcekli,color='red')
plt.plot(yeniveri_olcekli,svr_reg.predict(yeniveri_olcekli),color='blue')

plt.show()

print('SVR R2 degeri')
print(r2_score(maaslar_olcekli, svr_reg.predict(yeniveri_olcekli)))


#Decision Tree Regresyon
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(yeniveri,maaslar)
Z = yeniveri + 0.5
K = yeniveri - 0.4

plt.scatter(yeniveri,maaslar, color='red')
plt.plot(yeniveri,r_dt.predict(yeniveri), color='blue')
plt.plot(yeniveri,r_dt.predict(Z),color='green')
plt.plot(yeniveri,r_dt.predict(K),color='yellow')
plt.show()

print('Decision Tree R2 degeri')
print(r2_score(maaslar, r_dt.predict(yeniveri)))


#Random Forest Regresyonu estimators kaç alt ağaç olsun 30 dedim
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 30,random_state=0)
rf_reg.fit(yeniveri,maaslar.ravel())


plt.scatter(yeniveri,maaslar,color='red')
plt.plot(yeniveri,rf_reg.predict(yeniveri),color='blue')
plt.plot(yeniveri,rf_reg.predict(Z),color='green')
plt.plot(yeniveri,r_dt.predict(K),color='yellow')
plt.show()


print('Random Forest R2 degeri')

print(r2_score(maaslar, rf_reg.predict(yeniveri)))
print(r2_score(maaslar, rf_reg.predict(K)))
print(r2_score(maaslar, rf_reg.predict(Z)))

#Ozet R2 değerleri
print('-----------------------')
print('Linear R2 degeri')
print(r2_score(maaslar, lin_reg.predict(yeniveri)))

print('Polynomial R2 degeri')
print(r2_score(maaslar, lin_reg2.predict(poly_reg.fit_transform(yeniveri))))

print('SVR R2 degeri')
print(r2_score(maaslar_olcekli, svr_reg.predict(yeniveri_olcekli)))


print('Decision Tree R2 degeri')
print(r2_score(maaslar, r_dt.predict(yeniveri)))

print('Random Forest R2 degeri')
print(r2_score(maaslar, rf_reg.predict(yeniveri)))














