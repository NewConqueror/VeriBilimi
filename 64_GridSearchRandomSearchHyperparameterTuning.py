from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np

"""
Bu kod, Iris veri seti üzerinde üç farklı makine öğrenimi algoritması (K-Nearest Neighbors,
Decision Tree, ve Support Vector Machine) için hiperparametre optimizasyonu yapmaktadır. 
Kod, Grid Search ve Randomized Search tekniklerini kullanarak en iyi hiperparametre 
kombinasyonlarını bulmayı amaçlar.

knn_param_grid = {"n_neighbors": np.arange(2, 31)}: KNN için n_neighbors (komşu sayısı) 
parametresinin 2'den 30'a kadar olan değerlerini içeren bir hiperparametre aralığı belirler.
"""

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


"""
GridSearchCV(knn, knn_param_grid): KNN için tüm hiperparametre kombinasyonlarını dener ve 
en iyi kombinasyonu bulur.
RandomizedSearchCV(knn, knn_param_grid, n_iter=10): Hiperparametre kombinasyonlarının 
rastgele bir alt kümesini dener ve en iyi kombinasyonu bulur. 
n_iter=10, 10 farklı hiperparametre kombinasyonunu deneyeceği anlamına gelir.
"""

# KNN
knn = KNeighborsClassifier()
knn_param_grid = {"n_neighbors": np.arange(2, 31)}
knn_grid_search = GridSearchCV(knn, knn_param_grid)
knn_grid_search.fit(X_train, y_train)
print("KNN Grid Search Best Parameters: ", knn_grid_search.best_params_)
print("KNN Grid Search Best Accuracy: ", knn_grid_search.best_score_)

knn_random_search = RandomizedSearchCV(knn, knn_param_grid, n_iter=10)
knn_random_search.fit(X_train, y_train)
print("KNN Random Search Best Parameters: ", knn_random_search.best_params_)
print("KNN Random Search Best Accuracy: ", knn_random_search.best_score_)
print()

# DT
tree = DecisionTreeClassifier()
tree_param_grid = {"max_depth": [3, 5, 7],
                   "max_leaf_nodes": [None, 5, 10, 20, 30, 50]}

"""
tree_param_grid: Karar ağacı için max_depth (maksimum derinlik) ve 
max_leaf_nodes (maksimum yaprak düğüm sayısı) parametreleri için aralıklar belirler.
"""

tree_grid_search = GridSearchCV(tree, tree_param_grid)
tree_grid_search.fit(X_train, y_train)
print("DT Grid Search Best Parameters: ", tree_grid_search.best_params_)
print("DT Grid Search Best Accuracy: ", tree_grid_search.best_score_)

tree_random_search = RandomizedSearchCV(tree, tree_param_grid, n_iter=10)
tree_random_search.fit(X_train, y_train)
print("DT Random Search Best Parameters: ", tree_random_search.best_params_)
print("DT Random Search Best Accuracy: ", tree_random_search.best_score_)
print()

"""
Karar ağacı için Grid Search ve Randomized Search ile en iyi hiperparametreleri bulur.
"""

# SVM
svm = SVC()
svm_param_grid = {"C":[0.1, 1, 10, 100],
                  "gamma": [0.1, 0.01, 0.001, 0.0001]}


"""
svm_param_grid: SVM için C (ceza parametresi) ve gamma (kernel koefisiyenti) 
parametreleri için aralıklar belirler.
"""

svm_grid_search = GridSearchCV(svm, svm_param_grid)
svm_grid_search.fit(X_train, y_train)
print("SVM Grid Search Best Parameters: ", svm_grid_search.best_params_)
print("SVM Grid Search Best Accuracy: ", svm_grid_search.best_score_)

svm_random_search = RandomizedSearchCV(svm, svm_param_grid, n_iter=10)
svm_random_search.fit(X_train, y_train)
print("SVM Random Search Best Parameters: ", svm_random_search.best_params_)
print("SVM Random Search Best Accuracy: ", svm_random_search.best_score_)

"""
SVM için Grid Search ve Randomized Search ile en iyi hiperparametreleri bulur.

Bu kod, Iris veri seti üzerinde KNN, Karar Ağacı ve SVM sınıflandırıcıları için 
en iyi hiperparametreleri bulmayı amaçlar. Grid Search tüm olası kombinasyonları denerken, 
Randomized Search rastgele seçilmiş bir kombinasyon alt kümesini dener. 
Kodun çıktıları, her model için en iyi hiperparametreleri ve bu parametrelerle elde edilen 
en iyi doğruluk skorlarını gösterir.
"""