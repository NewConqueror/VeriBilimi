import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


"""
Bu kod iki farklı bölümden oluşuyor. İlk bölümde, iki bağımsız değişkenli bir 
doğrusal regresyon modeli oluşturulup, eğitiliyor ve görselleştiriliyor. 
İkinci bölümde ise Diabetes (şeker hastalığı) veri seti üzerinde çok değişkenli 
doğrusal regresyon modeli uygulanıyor ve modelin performansı ölçülüyor. 


X: 100 tane veri noktası oluşturuluyor, her biri iki özellik (X1 ve X2) içeriyor.
coef: Doğrusal regresyon katsayıları [3, 5] olarak belirlenmiş.
y: y, iki bağımsız değişkene bağlı olarak hesaplanır. np.dot(X, coef) ifadesi, 
X matrisinin coef katsayılarıyla çarpımını hesaplar ve her bir veri noktası için y değerini 
oluşturur. Ayrıca y'ye biraz rastgele gürültü eklenir (np.random.rand(100)).

3D Scatter Plot: Veri noktaları 3D bir grafikte görselleştirilir. 
X1 ve X2 eksenlerinde bağımsız değişkenler (özellikler), y ekseninde ise 
hedef değişken gösterilir.
"""

# y = a0 + a1x -> linear regression
# y = a0 + a1x1 + a2x2 + ... + anxn -> multi variable linear regression
# y = a0 + a1x1 + a2x2 

X = np.random.rand(100, 2)
coef = np.array([3, 5])
# y = 0 + np.dot(X, coef)
y = np.random.rand(100) + np.dot(X, coef)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection = "3d")
# ax.scatter(X[:, 0], X[:, 1], y)
# ax.set_xlabel("X1")
# ax.set_ylabel("X2")
# ax.set_zlabel("y")

lin_reg = LinearRegression()
lin_reg.fit(X, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:, 0], X[:, 1], y)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")

"""
x1, x2: meshgrid kullanılarak 3D yüzey için bir grid oluşturuluyor. 
Bu, her bir X1 ve X2 değeri kombinasyonu için bir y yüzey tahmini yapmak için kullanılır.
y_pred: Modelin bu grid noktalarındaki tahminleri hesaplanır.
plot_surface: Modelin tahmin ettiği yüzey, grafikte yarı saydam olarak (alpha=0.3) çizilir. 
Bu yüzey, modelin öğrendiği doğrusal düzlemi temsil eder.
"""

x1, x2 = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
y_pred = lin_reg.predict(np.array([x1.flatten(), x2.flatten()]).T)
ax.plot_surface(x1, x2, y_pred.reshape(x1.shape), alpha= 0.3)
plt.title("multi variable linear regression")

"""
lin_reg.coef_: Modelin öğrendiği eğim katsayıları (a1, a2).
lin_reg.intercept_: Modelin öğrendiği kesim noktası (a0).
"""

print("Katsayilar: ", lin_reg.coef_)
print("Kesisim: ", lin_reg.intercept_)

# %%
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

diabetes = load_diabetes()

X = diabetes.data 
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print("rmse: ", rmse)

"""
RMSE (Root Mean Squared Error): Modelin tahmin hatalarının karelerinin ortalaması alınarak 
hataların büyüklüğü ölçülür. squared=False ifadesi, hataların karekökünü alarak 
RMSE'yi hesaplar.
RMSE Değeri: Modelin test setindeki tahminlerinin ortalama hatasını ifade eder.


İlk Bölüm: İki bağımsız değişkenli bir doğrusal regresyon modeli oluşturulup eğitilir 
ve bu modelin öğrenme sonuçları 3D grafikte görselleştirilir.
Modelin katsayıları ve kesim noktası bulunur.
İkinci Bölüm: Diabetes veri seti üzerinde çok değişkenli doğrusal regresyon uygulanır, 
modelin performansı RMSE ile değerlendirilir ve sonuçlar görselleştirilir.
"""