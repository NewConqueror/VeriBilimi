from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

"""
Bu kod, sklearn kütüphanesini kullanarak Digits (Rakamlar) veri seti üzerinde bir 
Destek Vektör Makinesi (SVM) modeli oluşturur ve bu modelin performansını değerlendirir. 
Kodun ayrıntılı açıklaması şu şekildedir:

Veri Seti: Kod, load_digits() fonksiyonu ile Digits veri setini yükler. Bu veri seti, 
el yazısı ile yazılmış rakamların (0-9) dijital görüntülerini içerir. 
Her görüntü, 8x8 piksel boyutunda siyah beyaz bir görüntü olup, her piksel 0'dan 16'ya 
kadar bir gri ton değeri alır.


Görselleştirme: Kod, ilk 10 rakamın (görselin) 8x8 piksel görüntülerini 
bir matris düzeninde (2 satır, 5 sütun) gösterir.
ax.imshow(): Görüntüleri binary renk haritasıyla gösterir, bu da gri tonlarını temsil eder.
ax.set_title(): Her görüntünün üstüne o görüntünün temsil ettiği rakamı (target) yazar.
plt.show(): Bu fonksiyon, grafiği ekranda gösterir.

X değişkeni, her görüntüyü 64 (8x8) boyutunda bir özellik vektörüne çevirir.

"""

digits = load_digits()

fig, axes = plt.subplots(nrows = 2, ncols = 5, figsize = (10,5),
                         subplot_kw = {"xticks":[], "yticks":[]})

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = "binary", interpolation = "nearest")
    ax.set_title(digits.target[i])
    
plt.show()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_clf = SVC(kernel = "linear", random_state = 42)
svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_test)

print(classification_report(y_test, y_pred))

"""
SVM Modeli:
SVC(kernel = "linear"): SVC sınıfı ile bir SVM sınıflandırıcı oluşturulur. 
kernel="linear", doğrusal bir çekirdek (linear kernel) kullanılacağını belirtir.

Model Değerlendirmesi: classification_report() fonksiyonu, modelin performansını çeşitli 
metriklerle (precision, recall, f1-score) değerlendirir ve sonuçları ekrana yazdırır. 
Bu metrikler:
Precision: Modelin pozitif olarak sınıflandırdığı örneklerin ne kadarının gerçekten pozitif 
olduğunu gösterir.
Recall: Gerçek pozitiflerin ne kadarının model tarafından doğru bir şekilde pozitif olarak 
sınıflandırıldığını gösterir.
F1-Score: Precision ve recall'un harmonik ortalamasıdır ve dengesiz veri setlerinde 
önemlidir.
"""

"""
Bu kod, Digits veri seti üzerinde bir SVM modelini kullanarak rakamları sınıflandırır. 
Model, doğrusal bir çekirdek kullanılarak eğitilir ve test verisi üzerinde değerlendirilir. 
Sonuç olarak, modelin performansı precision, recall, f1-score gibi metriklerle analiz edilir 
ve raporlanır.
"""