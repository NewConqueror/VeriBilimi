# https://archive.ics.uci.edu/dataset/45/heart+disease

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

"""
Bu kod, Heart Disease (Kalp Hastalığı) veri setini kullanarak bir Lojistik Regresyon modeli 
oluşturur ve bu modelin doğruluğunu (accuracy) hesaplar. 
Kodun ayrıntılı açıklaması aşağıdaki gibidir:

Veri Seti: Kod, UCI makine öğrenmesi veri tabanından Heart Disease veri setini ucimlrepo 
modülü ile yükler. Bu veri seti, kalp hastalığı riski olan bireylerle ilgili çeşitli 
özellikler (örneğin yaş, cinsiyet, kolesterol seviyesi) içerir ve hastalığın olup olmadığını 
belirten bir hedef değişken (target) içerir.
"""

heart_disease = fetch_ucirepo(name = "heart_disease")

df = pd.DataFrame(data = heart_disease.data.features)

df["target"] = heart_disease.data.targets

# drop missing value
if df.isna().any().any():
    df.dropna(inplace = True)
    print("nan")
    
X = df.drop(["target"], axis = 1).values
y = df.target.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

log_reg = LogisticRegression(penalty="l2", C=1, solver="lbfgs", max_iter=100)
log_reg.fit(X_train, y_train)

accuracy = log_reg.score(X_test, y_test)
print("Logistic Regression Acc:", accuracy)

"""
Lojistik Regresyon Modeli: LogisticRegression sınıfı kullanılarak bir lojistik regresyon 
modeli oluşturulur. 
Modelin parametreleri:
penalty="l2": L2 normunu (Ridge) kullanarak regularization (düzenleme) yapar. 
Bu, aşırı uyum (overfitting) riskini azaltmaya yardımcı olur.
C=1: Regularization teriminin tersini kontrol eder. 
Düşük C değerleri daha güçlü regularization sağlar.
solver="lbfgs":Küçük ve orta ölçekli veri setleri için uygun bir optimizasyon algoritmasıdır.
max_iter=100: Maksimum 100 iterasyon yapılır.


Bu kod, UCI'den alınan kalp hastalığı veri setini kullanarak bir lojistik regresyon modeli 
oluşturur ve test seti üzerindeki doğruluk oranını hesaplar. Bu doğruluk oranı, 
modelin kalp hastalığı olup olmadığını tahmin etmede ne kadar başarılı olduğunu gösterir.
"""