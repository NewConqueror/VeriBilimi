# sklearn: ML Library
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

"""
Bu kod, iki farklı makine öğrenmesi probleminde K-Nearest Neighbors (KNN) algoritmasını 
kullanarak sınıflandırma ve regresyon işlemlerini gerçekleştiriyor. 
Kod iki ana bölümden oluşuyor:
"""

# (1) Veri Seti İncelenmesi
cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
df["target"] = cancer.target

# (2) Makine Öğrenmesi Modelinin Seçilmesi – KNN Sınıflandırıcı
# (3) Modelin Train Edilmesi
X = cancer.data # features
y = cancer.target # target

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# olceklendirme 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# knn modeli olustur ve train et
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, y_train) # fit fonksiyonu verimizi (samples + target) kullanarak knn algoritmasini egitir

# (4) Sonuçların Değerlendirilmesi: test
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Dogruluk:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("confusion_matrix:")
print(conf_matrix)


# (5) Hiperparametre Ayarlaması
"""
    KNN: Hyperparameter = K
        K: 1,2,3 ... N
        Accuracy: %A, %B, %C ....
"""
accuracy_values = []
k_values = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)

plt.figure()
plt.plot(k_values, accuracy_values, marker = "o", linestyle = "-")
plt.title("K degerine gore dogruluk")
plt.xlabel("K degeri")
plt.ylabel("Dogruluk")
plt.xticks(k_values)
plt.grid(True)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

"""
40 adet rastgele sayı üretilir ve bu sayılar, bir sinüs fonksiyonuna uygulanarak hedef 
değerler (y) elde edilir.
Her 5. veri noktasına rastgele bir gürültü eklenir.
0 ile 5 arasında 500 eşit aralıkta nokta oluşturulur.
"""

X = np.sort(5 * np.random.rand(40, 1), axis = 0) # features
y = np.sin(X).ravel() # target

# plt.scatter(X, y)

# add noise
y[::5] += 1 * (0.5 - np.random.rand(8)) 

# plt.scatter(X, y)
T = np.linspace(0, 5, 500)[:, np.newaxis]

for i, weight in enumerate(["uniform", "distance"]):

    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
    y_pred = knn.fit(X,y).predict(T)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color = "green", label = "data")
    plt.plot(T, y_pred, color = "blue", label = "prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN Regressor weights = {}".format(weight))

plt.tight_layout()
plt.show()

"""
Regresyon Modeli ve Sonuçların Görselleştirilmesi: 
KNN Regressor modeli uniform ve distance ağırlıklandırma seçenekleri ile iki kez uygulanır. 
Tahmin sonuçları ve orijinal veriler grafikte gösterilir.

uniform: Her komşunun oyu eşit ağırlıkta alınır.
distance: Komşuların ağırlıkları uzaklıklarına göre belirlenir; 
daha yakın olan komşulara daha fazla ağırlık verilir.
Bu kodun tümü, KNN algoritmasının sınıflandırma ve regresyon problemlerinde 
nasıl çalıştığını göstermek için kullanılır.
"""