from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error


"""
Bu kod, Diabetes veri seti üzerinde Ridge ve Lasso regresyon modelleri için 
en iyi hiperparametreleri bulmayı ve bu modellerin test veri seti üzerindeki performansını 
değerlendirmeyi amaçlamaktadır. Kod, Grid Search kullanarak her iki modelin de en iyi alpha 
parametresini belirler ve bu parametrelerle eğitilen modellerin ortalama kare hata (MSE) 
metriklerini hesaplar.
"""

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge
ridge = Ridge()
ridge_param_grid = {"alpha": [0.1, 1, 10, 100]}

ridge_grid_search = GridSearchCV(ridge, ridge_param_grid, cv = 5)
ridge_grid_search.fit(X_train, y_train)
print("Ridge en iyi parameters: ", ridge_grid_search.best_params_)
print("Ridge en iyi score: ", ridge_grid_search.best_score_)

best_ridge_model = ridge_grid_search.best_estimator_
y_pred_ridge = best_ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
print("ridge_mse: ",ridge_mse)
print()


"""
Ridge(): Ridge regresyon modelini oluşturur. Ridge regresyonu, L2 normuna dayalı bir ceza 
terimi ekleyerek regresyon modelini düzenler.
ridge_param_grid: alpha hiperparametresi için dört farklı değer belirler (0.1, 1, 10, 100). 
alpha, modelin ceza teriminin gücünü kontrol eder.
GridSearchCV(ridge, ridge_param_grid, cv=5): 5 katlı çapraz doğrulama kullanarak 
Grid Search uygular. En iyi alpha parametresini bulur.
ridge_grid_search.fit(X_train, y_train): Eğitim verisi üzerinde Grid Search işlemini 
gerçekleştirir.
best_estimator_: En iyi parametrelerle eğitilen Ridge modelini döndürür.
mean_squared_error(y_test, y_pred_ridge): Test veri seti üzerindeki tahminler ile gerçek 
değerler arasındaki ortalama kare hatayı hesaplar.
"""


# Lasso
lasso = Lasso()
lasso_param_grid = {"alpha": [0.1, 1, 10, 100]}

lass_grid_search = GridSearchCV(lasso, lasso_param_grid, cv = 5)
lass_grid_search.fit(X_train, y_train)
print("Lasso en iyi parameters: ", lass_grid_search.best_params_)
print("Lasso en iyi score: ", lass_grid_search.best_score_)

best_lasso_model = lass_grid_search.best_estimator_
y_pred_lasso = best_lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
print("lasso_mse: ",lasso_mse)


"""
Lasso(): Lasso regresyon modelini oluşturur. Lasso regresyonu, L1 normuna dayalı bir ceza 
terimi ekleyerek regresyon modelini düzenler ve aynı zamanda bazı özelliklerin 
katsayılarını sıfıra indirerek modelde otomatik özellik seçimi yapar.
lasso_param_grid: alpha hiperparametresi için aynı dört değeri belirler.
GridSearchCV(lasso, lasso_param_grid, cv=5): Lasso modeli için Grid Search uygular ve 
en iyi alpha değerini bulur.
mean_squared_error(y_test, y_pred_lasso): Test veri seti üzerindeki Lasso tahminleri için 
ortalama kare hatayı hesaplar.

Ridge ve Lasso regresyonları için en iyi alpha parametreleri Grid Search ile belirlenir.
Ridge ve Lasso modelleri, belirlenen en iyi alpha parametreleri ile eğitilir ve 
test veri seti üzerindeki performansları ortama kare hata (MSE) metriği ile değerlendirilir.
Kod, her iki modelin de en iyi hiperparametre kombinasyonunu ve bu kombinasyonla elde edilen 
doğruluk skorlarını çıktılar. Ridge ve Lasso regresyonları karşılaştırılarak hangi modelin 
daha iyi performans gösterdiği değerlendirilebilir.
"""