from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

"""
Bu kod iki farklı bölümden oluşuyor. İlk bölümde basit bir doğrusal regresyon modeli 
bir yapay veri seti üzerinde uygulanıyor ve görselleştiriliyor. 
İkinci bölümde ise, bir gerçek dünya veri seti olan Diabetes (şeker hastalığı) 
veri seti üzerinde doğrusal regresyon modeli uygulanıyor ve modelin performansı ölçülüyor.

X: 100 tane 1 boyutlu rastgele sayılar (0 ile 1 arasında) oluşturuluyor.
y: Doğrusal bir fonksiyon olarak y = 3 + 4x + hata oluşturuluyor. 
Burada hata, np.random.rand(100, 1) ifadesiyle rastgele küçük gürültü eklenerek oluşturuluyor.


"""

# veri olustur
X = np.random.rand(100,1)
y = 3 + 4 * X + np.random.rand(100,1) # y = 3 + 4x 

# plt.scatter(X,y)

lin_reg = LinearRegression()
lin_reg.fit(X, y)

plt.figure()
plt.scatter(X,y)
plt.plot(X, lin_reg.predict(X), color = "red", alpha = 0.7)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Lineer Regresyon")


"""
a1: Regresyon katsayısı (eğim). coef_ ile modelin eğimi elde edilir.
a0: Regresyon sabiti (kesim noktası). intercept_ ile modelin kesim noktası elde edilir.
Bu değerler ekranda yazdırılır.

Bu döngü, 100 kez yeşil renkte (her biri aynı olan) doğrusal çizgiyi grafikte çiziyor. 
Ancak bu çizgilerin hepsi aynı olduğu için görsel olarak sadece bir yeşil çizgi olarak 
görünür. Bu adım, modeli ve çizgiyi birden çok kez çizdiriyor.
"""

# y = 3 + 4x  -> y = a0 + a1x
a1 = lin_reg.coef_[0][0] 
print("a1: ",a1)

a0 = lin_reg.intercept_[0]
print("a0: ",a0)

for i in range(100):
    y_ = a0 + a1 * X
    plt.plot(X, y_, color = "green", alpha = 0.7)

# %%
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import numpy as np

diabetes = load_diabetes()

diabetes_X, diabetes_y = load_diabetes(return_X_y = True)

"""
diabetes_X: Orijinal veri setinde birçok özellik varken, burada sadece üçüncü sütun 
(indeks 2) seçiliyor. np.newaxis bu sütunun 2D (n x 1) bir vektör olarak temsil edilmesini 
sağlar.
"""

diabetes_X = diabetes_X[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-20] 
diabetes_X_test = diabetes_X[-20:] 

diabetes_y_train = diabetes_y[:-20] 
diabetes_y_test = diabetes_y[-20:] 

lin_reg = LinearRegression()

lin_reg.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = lin_reg.predict(diabetes_X_test)

mse = mean_squared_error(diabetes_y_test, diabetes_y_pred)
print("mse: ", mse)
r2 = r2_score(diabetes_y_test, diabetes_y_pred)
print("r2: ", r2)

plt.scatter(diabetes_X_test, diabetes_y_test, color = "black")
plt.plot(diabetes_X_test, diabetes_y_pred, color = "blue")

"""
MSE (Mean Squared Error): Modelin tahminlerinin hatalarının karelerinin ortalaması olarak 
hesaplanır. Bu, hatanın büyüklüğünü gösterir.
R² Skoru: Modelin performansını değerlendirir. 1'e yakın bir R² skoru, modelin veri ile 
çok iyi uyum sağladığını gösterir.


İlk Bölüm: Yapay bir veri seti üzerinde basit bir doğrusal regresyon modeli oluşturulur, 
eğitilir ve sonuçlar görselleştirilir. Modelin katsayıları bulunur ve grafik üzerinde 
gösterilir.
İkinci Bölüm: Diabetes veri seti üzerinde doğrusal regresyon uygulanır, modelin performansı 
MSE ve R² ile değerlendirilir ve sonuçlar görselleştirilir.
"""