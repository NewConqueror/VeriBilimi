import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

"""
Bu kod, polinomiyal regresyonu (polynomial regression) uygulamak için tasarlanmıştır. 
Polinomiyal regresyon, bağımsız değişkenler ve hedef değişken arasında doğrusal olmayan 
ilişkileri modellemek için kullanılır

X: 0 ile 4 arasında rastgele 100 veri noktası oluşturur. np.random.rand(100, 1) 
100 adet rastgele değer üretir ve bunları 4 ile çarpar.
y: y = 2 + 3*X^2 şeklinde bir ilişki ile hesaplanır. Burada X^2 terimi doğrusal olmayan 
bir ilişkiyi temsil eder. Ayrıca, rastgele bir gürültü eklenir (2 * np.random.rand(100, 1)), 
bu da gerçek ilişkiden küçük sapmalar oluşturur.

PolynomialFeatures: Özellikleri belirtilen derecede polinomiyal özellikler olarak 
dönüştürür. Burada degree=2 ile ikinci dereceden polinomiyal özellikler oluşturulur.

fit_transform: Bu yöntem X veri setini alır ve ikinci dereceden polinomiyal özellikleri 
oluşturur. Bu özellikler X'in kendisi ve X^2'yi içerir.
"""

X = 4 * np.random.rand(100, 1)
y = 2 + 3*X**2 + 2 * np.random.rand(100, 1) # y = 2 + 3x^2

# plt.scatter(X, y)

"""
 1. derecen denklem => y = a0 + a1x => lineer regresyon
"""
poly_feat = PolynomialFeatures(degree=2)
X_poly = poly_feat.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

plt.scatter(X, y, color = "blue")

X_test = np.linspace(0, 4, 100).reshape(-1, 1)
X_test_poly = poly_feat.transform(X_test)
y_pred = poly_reg.predict(X_test_poly)

plt.plot(X_test, y_pred, color = "red")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polinom Refresyon Modeli")

"""
Gerçek veri noktaları X ve y'yi mavi renkli noktalar olarak gösterir.
Polinomiyal regresyon modelinin tahmin ettiği eğriyi kırmızı renkli çizgi olarak çizer.

Veri Oluşturma: Doğrusal olmayan y = 2 + 3x^2 formülü ile rastgele bir veri seti oluşturulur.
Polinomiyal Özellikler: Özellikler, ikinci dereceden (quadratic) polinomiyal terimler 
eklenerek dönüştürülür.
Model Eğitimi: Polinomiyal özellikler kullanılarak doğrusal regresyon modeli eğitilir.
Görselleştirme:Gerçek veri noktaları ve modelin tahmin ettiği polinomiyal eğri görselleştirilir.

Bu kod, doğrusal regresyonun polinomiyal genişletilmesini ve böylece daha 
karmaşık ilişkileri modelleme yeteneğini gösterir.
"""