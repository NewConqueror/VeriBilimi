from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

"""
Bu kod, Elastic Net regresyon modelini kullanarak Diabetes veri seti üzerinde en iyi 
hiperparametreleri bulmayı ve modelin test veri seti üzerindeki performansını 
değerlendirmeyi amaçlamaktadır. Elastic Net, Ridge ve Lasso regresyonlarının bir 
kombinasyonu olarak çalışır ve hem L1 (Lasso) hem de L2 (Ridge) cezalarını kullanarak 
modelin düzenlenmesini sağlar.
"""

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

elastic_net = ElasticNet()
elastic_net_param_grid = {"alpha": [0.1, 1, 10, 100],
                          "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]} # l1 or l2 penalty ratio


"""
ElasticNet(): Elastic Net regresyon modelini oluşturur. Elastic Net, Lasso ve Ridge 
regresyonlarının avantajlarını birleştirir.
elastic_net_param_grid: Hiperparametreler için bir ızgara (grid) belirler:
alpha: Modelin ceza teriminin gücünü kontrol eder (L1 ve L2 cezalarının toplamını).
l1_ratio: L1 ve L2 cezaları arasındaki oranı belirler. 1'e yakın değerler Lasso, 
0'a yakın değerler Ridge regresyonuna daha fazla ağırlık verir.
"""

elastic_net_grid_search = GridSearchCV(elastic_net, elastic_net_param_grid, cv = 5)
elastic_net_grid_search.fit(X_train, y_train)
print("En iyi parameter: ",elastic_net_grid_search.best_params_)
print("En iyi score: ",elastic_net_grid_search.best_score_)

"""
GridSearchCV(elastic_net, elastic_net_param_grid, cv=5): 5 katlı çapraz doğrulama kullanarak 
Elastic Net modeli için Grid Search uygular. Bu yöntem, belirlenen hiperparametre 
ızgarasındaki tüm kombinasyonları dener ve en iyi alpha ve l1_ratio değerlerini bulur.
"""

best_elastic_net_model = elastic_net_grid_search.best_estimator_
y_pred = best_elastic_net_model.predict(X_test)

"""
best_estimator_: En iyi parametrelerle eğitilen Elastic Net modelini döndürür.
"""

mse = mean_squared_error(y_test, y_pred)
print("mse: ",mse)

"""
Elastic Net regresyonu için en iyi alpha ve l1_ratio parametrelerini Grid Search kullanarak 
belirler.
En iyi model, bu parametrelerle eğitilir ve test veri seti üzerindeki performansı ortalama 
kare hata (MSE) metriği ile değerlendirilir.
Kod, en iyi hiperparametre kombinasyonunu ve bu kombinasyonla elde edilen doğruluk skorunu 
çıktılar. Test veri setindeki ortalama kare hata değeri de çıktı olarak gösterilir. 
Bu, modelin performansını değerlendirmenize yardımcı olur.
"""