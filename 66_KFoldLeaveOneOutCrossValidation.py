from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

"""
Bu kod, Iris veri seti üzerinde Karar Ağacı (Decision Tree) sınıflandırıcısı için iki farklı 
çapraz doğrulama yöntemi kullanarak en iyi hiperparametreleri bulmayı amaçlamaktadır. 
Bu iki çapraz doğrulama yöntemi K-Fold ve Leave-One-Out (LOO) olarak adlandırılır. 
Kod, her bir yöntemle hiperparametre araması yapmak için Grid Search kullanır ve 
en iyi parametreleri ve bu parametrelerle elde edilen doğruluk (accuracy) skorlarını yazdırır.
"""

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier()
tree_param_dist = {"max_depth": [3, 5, 7]}

"""
tree_param_dist: Karar ağacı için max_depth (maksimum derinlik) hiperparametresi için 
üç farklı seçenek belirler: 3, 5, ve 7.
"""

# KFOLD Grid Search
kf = KFold(n_splits = 10)
tree_grid_search_kf = GridSearchCV(tree, tree_param_dist, cv = kf)
tree_grid_search_kf.fit(X_train, y_train)
print("KF En iyi paramter: ", tree_grid_search_kf.best_params_)
print("KF En iyi acc: ", tree_grid_search_kf.best_score_)

"""
KFold(n_splits=10): Veriyi 10 parçaya böler ve her bir parça test seti olarak kullanılırken, 
kalan 9 parça eğitim seti olarak kullanılır (10 katlı çapraz doğrulama).
GridSearchCV(tree, tree_param_dist, cv=kf): K-Fold çapraz doğrulama kullanarak 
Grid Search uygular. Hiperparametre kombinasyonlarını dener ve en iyisini bulur.
"""

# LOO
loo = LeaveOneOut()
tree_grid_search_loo = GridSearchCV(tree, tree_param_dist, cv = loo)
tree_grid_search_loo.fit(X_train, y_train)
print("LOO En iyi paramter: ", tree_grid_search_loo.best_params_)
print("LOO En iyi acc: ", tree_grid_search_loo.best_score_)

"""
LeaveOneOut(): Verideki her bir örnek tek başına test seti olarak kullanılırken, 
diğer tüm örnekler eğitim seti olarak kullanılır (LOO çapraz doğrulama).
GridSearchCV(tree, tree_param_dist, cv=loo): Leave-One-Out çapraz doğrulama kullanarak 
Grid Search uygular.
"""

"""
Kod,Iris veri seti üzerinde karar ağacı algoritması için en iyi max_depth hiperparametresini 
bulmayı amaçlamaktadır.
K-Fold ve Leave-One-Out çapraz doğrulama yöntemleri ile Grid Search kullanılarak, 
her iki yöntemle de en iyi hiperparametre kombinasyonu ve elde edilen en yüksek doğruluk 
skorları hesaplanır ve yazdırılır.
K-Fold, veriyi belirli sayıda katmana bölerken, Leave-One-Out her bir veri noktasını 
ayrı bir test seti olarak kullanır, bu da daha hassas bir değerlendirme sağlar.
"""