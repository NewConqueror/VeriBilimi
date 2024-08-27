from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

"""
Bu kodlar, MNIST ve Iris veri setlerine Lineer Diskriminant Analizi (LDA) ve 
Ana Bileşen Analizi (PCA) uygulayarak veri setlerini iki boyutlu bir alanda görselleştirir. 
Bu sayede verilerin dağılımı ve sınıflar arası farkların incelenmesi sağlanır.

fetch_openml("mnist_784", version=1): MNIST veri setini OpenML'den indirir. Bu veri seti, 
el yazısı rakamların (0-9) gri tonlamalı görüntülerini içerir.
X = mnist.data: Veri setindeki görüntülerin piksel değerleri.
y = mnist.target.astype(int): Etiketler (hangi rakamın olduğu bilgisi). 
Veriler tam sayıya çevrilir.
lda = LinearDiscriminantAnalysis(n_components=2): LDA modeli, iki bileşene indirgenecek 
şekilde tanımlanır.
X_lda = lda.fit_transform(X, y): LDA modeli, MNIST veri setine uygulanır ve iki boyutlu bir 
veri kümesi elde edilir.
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap="tab10", alpha=0.6): LDA ile indirgenmiş 
veriler, iki boyutlu bir grafikte görselleştirilir. Renkler, verilerin hangi rakama 
ait olduğunu gösterir.
"""

mnist = fetch_openml("mnist_784", version=1)

X = mnist.data
y = mnist.target.astype(int)

lda = LinearDiscriminantAnalysis(n_components=2)

X_lda = lda.fit_transform(X, y)

plt.figure()
plt.scatter(X_lda[:, 0], X_lda[:, 1], c = y, cmap = "tab10", alpha = 0.6)
plt.title("LDA of MNIST Dataset")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.colorbar(label="Digits")

# %% LDA vs PCA
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

iris = load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

colors = ["red", "blue", "green"]

"""
plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.8, label=target_name): 
PCA ile indirgenmiş veriler, iki boyutlu bir grafikte görselleştirilir. Renkler, 
farklı iris çiçeği türlerini gösterir.

plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], color=color, alpha=0.8, label=target_name): 
LDA ile indirgenmiş veriler, iki boyutlu bir grafikte görselleştirilir. Renkler, 
farklı iris çiçeği türlerini gösterir.

"""

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color = color, alpha = 0.8, label = target_name)
plt.legend()
plt.title("PCA of Iris Dataset")

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], color = color, alpha = 0.8, label = target_name)
plt.legend()
plt.title("LDA of Iris Dataset")

"""
LDA (Linear Discriminant Analysis): Veriler arasındaki sınıf farklarını maksimize eden bir 
projeksiyon yöntemidir. Bu, sınıfları daha iyi ayrıştırmak için verileri düşük boyutlu bir 
alana indirger.
PCA (Principal Component Analysis): Verinin toplam varyansını maksimize eden bir projeksiyon 
yöntemidir.Bu,veri setindeki en fazla bilgi içeren (en yüksek varyansa sahip) boyutları seçer.
Kod, hem LDA hem de PCA'nın sonuçlarını karşılaştırarak, her iki yöntemle elde edilen 
iki boyutlu grafiklerde verilerin nasıl ayrıldığını görmemizi sağlar. 
PCA, veri setindeki genel varyansı dikkate alırken, LDA sınıflar arasındaki farkları 
ön plana çıkarır. Bu nedenle, Iris veri setinde LDA, sınıfların daha net ayrıştığı bir 
görselleştirme sağlar.
"""