from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

"""
Bu kod, Iris veri setini kullanarak Naive Bayes (GaussianNB) sınıflandırıcı ile bir 
model oluşturur ve bu modelin performansını değerlendirir. 
Aşağıda adım adım ne yaptığı açıklanmıştır:

classification_report: Modelin performansını ölçmek için kullanılan bir rapor üretir.

Iris Veri Seti: Çiçek türlerini sınıflandırmak için kullanılan popüler bir veri setidir. 
Üç farklı çiçek türünü (Setosa, Versicolor, Virginica) temsil eden 150 örnek içerir. 
Her örnek, dört özellikten (sepal length, sepal width, petal length, petal width) oluşur.

"""

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

y_pred = nb_clf.predict(X_test)

print(classification_report(y_test, y_pred))

"""
GaussianNB: Naive Bayes sınıflandırıcısının Gaussian varyantını kullanır. 
Bu varyant, özelliklerin sürekli ve normal dağılım gösterdiği varsayımına dayanır.


Performans Değerlendirmesi:
classification_report: Bu fonksiyon, modelin performansını değerlendirmek için çeşitli 
metriklerle (precision, recall, f1-score, accuracy) bir rapor oluşturur:
Precision: Modelin pozitif olarak sınıflandırdığı örneklerin ne kadarının gerçekten pozitif 
olduğunu gösterir.
Recall: Gerçek pozitif örneklerin ne kadarının model tarafından doğru şekilde pozitif olarak 
sınıflandırıldığını gösterir.
F1-Score: Precision ve recall'un harmonik ortalamasıdır.
Accuracy: Modelin doğru tahmin oranıdır.


Bu kod, Iris veri seti üzerinde bir Gaussian Naive Bayes modeli oluşturur ve 
test seti üzerinde modelin performansını değerlendirir. 
Performans sonuçları, precision, recall, f1-score gibi metriklerle raporlanır. 
Gaussian Naive Bayes, verilerin normal dağılım gösterdiği varsayımına dayalıdır ve 
Iris veri seti gibi küçük ve basit veri setlerinde genellikle iyi performans gösterir.

"""