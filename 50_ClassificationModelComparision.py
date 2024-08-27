from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

"""
Bu kod, farklı sınıflandırma algoritmalarını çeşitli yapay veri kümeleri üzerinde 
uygulayarak, sınıflandırma performanslarını ve karar sınırlarını görselleştiren 
bir çalışma yapmaktadır.

sklearn.datasets: Yapay veri kümeleri oluşturmak için kullanılan fonksiyonları içerir.
sklearn.model_selection: Veri setini eğitim ve test setlerine bölmek için train_test_split 
fonksiyonunu sağlar.
sklearn.preprocessing: Veriyi standartlaştırmak için StandardScaler kullanılır.
sklearn.pipeline: Farklı aşamaları birleştirerek bir iş akışı oluşturmak için make_pipeline 
kullanılır.
sklearn.inspection: Karar sınırlarını görselleştirmek için DecisionBoundaryDisplay kullanılır.
matplotlib.pyplot: Grafikler çizmek için kullanılır.
numpy: Sayısal işlemler için kullanılır.
ListedColormap: Özelleştirilmiş renk haritaları oluşturmak için kullanılır.
"""

"""
make_classification: Bilgilendirici ve gereksiz özelliklerle bir sınıflandırma 
veri kümesi oluşturur.
X += 1.2 * np.random.uniform(size=X.shape): Verilere rastgele bir gürültü ekler.
Xy = (X, y): Veri ve etiketler bir tuple olarak saklanır.
"""

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class = 1, random_state = 42)
X += 1.2 * np.random.uniform(size = X.shape)
Xy = (X, y)
# plt.scatter(X[:, 0], X[:, 1], c = y)

# X, y = make_moons(noise = 0.2, random_state=42)
# plt.scatter(X[:, 0], X[:, 1], c = y)

# X, y = make_circles(noise = 0.1, factor = 0.3, random_state=42)
# plt.scatter(X[:, 0], X[:, 1], c = y)

"""
make_moons: Ay şeklinde iki sınıf verisi oluşturur, gürültü ekler.
make_circles: İç içe geçmiş iki daire şeklinde veri kümesi oluşturur.
"""

datasets = [Xy,
            make_moons(noise = 0.2, random_state=42),
            make_circles(noise = 0.1, factor = 0.3, random_state=42)]

fig = plt.figure(figsize = (6,9))
i = 1
for ds_cnt, ds in enumerate(datasets):
    X, y = ds

    ax = plt.subplot(len(datasets), 1, i)
    ax.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.coolwarm, edgecolors = "black")
    i += 1
    
plt.show()    
    
names = ["Nearest Neighbors", "Linear SVM", "Decision Tree", "Random Forest", "Naive Bayes"]

"""
Beş farklı sınıflandırıcı tanımlanır: KNN, SVM, Karar Ağacı, Rastgele Orman, Naive Bayes.
"""
classifiers = [KNeighborsClassifier(), 
               SVC(), 
               DecisionTreeClassifier(), 
               RandomForestClassifier(), 
               GaussianNB()]
    
fig = plt.figure(figsize = (6,9))    
i = 1
for ds_cnt, ds in enumerate(datasets):
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    cm_bright = ListedColormap(["darkred", "darkblue"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
        
    # plot training data
    ax.scatter(X_train[:,0], X_train[:,1], c = y_train, cmap = cm_bright, edgecolors = "black")
    
    # plot test data
    ax.scatter(X_train[:,0], X_train[:,1], c = y_train, cmap = cm_bright, edgecolors = "black", alpha = 0.6)
    
    i += 1
    
    for name, clf in zip(names, classifiers):
        
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test) # accuracy
        DecisionBoundaryDisplay.from_estimator(clf, X, cmap = plt.cm.RdBu, alpha = 0.7, ax = ax, eps = 0.5)
        
        # plot training data
        ax.scatter(X_train[:,0], X_train[:,1], c = y_train, cmap = cm_bright, edgecolors = "black")
        
        # plot test data
        ax.scatter(X_train[:,0], X_train[:,1], c = y_train, cmap = cm_bright, edgecolors = "black", alpha = 0.6)
        
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            X[:, 0].max() - 0.15,
            X[:, 1].min() - 0.35,
            str(score))
        
        i += 1
        
"""
Her veri kümesi için beş sınıflandırıcı eğitilir ve test edilir.
Standartlaştırma: Veriler StandardScaler ile standartlaştırılır.
Karar Sınırları: DecisionBoundaryDisplay kullanılarak her sınıflandırıcı için 
karar sınırları çizilir.
Sonuçların Görselleştirilmesi: Eğitim ve test verileri ile birlikte karar sınırları da 
grafiklerde gösterilir. Her grafikte, modelin doğruluk skoru da görüntülenir.
"""

"""
Bu kod, üç farklı yapay veri kümesi (linear, moons, circles) üzerinde beş farklı 
sınıflandırıcıyı (KNN, SVM, Karar Ağacı, Rastgele Orman, Naive Bayes) uygular. 
Kod,her bir sınıflandırıcının karar sınırlarını çizerek, hangi modelin hangi veri kümesinde 
nasıl performans gösterdiğini görselleştirir. Bu sayede, farklı algoritmaların belirli 
veri kümelerinde nasıl çalıştığını anlamak için iyi bir görsel analiz sunar.
"""