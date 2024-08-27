from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier


"""
Bu kod, Iris veri seti üzerinde Decision Tree (Karar Ağacı) sınıflandırıcısı için 
hiperparametre optimizasyonu yapmaktadır. Hiperparametre optimizasyonu için 
Grid Search yöntemi kullanılmaktadır ve her bir parametre kombinasyonu için k-katlı çapraz 
doğrulama (cross-validation) yapılmaktadır. Kod ayrıca her bir parametre kombinasyonu için 
elde edilen doğruluk (accuracy) sonuçlarını ayrıntılı olarak yazdırır.
"""


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DT
tree = DecisionTreeClassifier()
tree_param_grid = {"max_depth": [3, 5, 7],
                   "max_leaf_nodes": [None, 5, 10, 20, 30, 50]}

"""
tree_param_grid: Karar ağacı için max_depth (maksimum derinlik) ve 
max_leaf_nodes (maksimum yaprak düğüm sayısı) gibi hiperparametrelerin aralıklarını belirler.
"""

nb_cv = 3
tree_grid_search = GridSearchCV(tree, tree_param_grid, cv = nb_cv)
tree_grid_search.fit(X_train, y_train)
print("DT Grid Search Best Parameters: ", tree_grid_search.best_params_)
print("DT Grid Search Best Accuracy: ", tree_grid_search.best_score_)

"""
GridSearchCV(tree, tree_param_grid, cv = nb_cv): Grid Search yöntemi kullanılarak her bir 
hiperparametre kombinasyonu için çapraz doğrulama yapılır. cv = nb_cv parametresi, 
3 katlı çapraz doğrulama yapılacağını belirtir.
"""

for mean_score, params in zip(tree_grid_search.cv_results_["mean_test_score"], tree_grid_search.cv_results_["params"]):
    print(f"Ortalama test skoru: {mean_score}, Parametreler: {params}")

"""
cv_results_: Grid Search sonuçlarını içerir.
Bu döngü, her bir parametre kombinasyonu için elde edilen ortalama test skorunu ve ilgili 
parametre kombinasyonlarını yazdırır.
"""
    
cv_result = tree_grid_search.cv_results_
for i, params in enumerate((cv_result["params"])):
    print(f"Parametreler: {params}")
    for j in range(nb_cv):
        accuracy = cv_result[f"split{j}_test_score"][i]
        print(f"\tFold {j+1} - Accuracy: {accuracy}")
    

"""
Bu bölüm, her bir parametre kombinasyonu için tüm çapraz doğrulama fold'larının (katlarının) 
doğruluk skorlarını yazdırır.
Her bir parametre kombinasyonu için 3 farklı doğruluk skoru yazdırılır, 
çünkü cv=3 (3 katlı çapraz doğrulama) yapılmıştır.

Bu kod, Iris veri seti üzerinde karar ağacı algoritması için en iyi hiperparametreleri 
bulmayı amaçlar. Her bir hiperparametre kombinasyonu için 3 katlı çapraz doğrulama yapılır 
ve en iyi sonuçlar yazdırılır. Ayrıca, tüm parametre kombinasyonları için ayrı ayrı 
elde edilen doğruluk skorları da detaylı olarak sunulur.

"""