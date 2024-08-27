from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

"""
Bu kod, MNIST veri setini indirir ve t-Distributed Stochastic Neighbor Embedding 
(t-SNE) algoritmasını kullanarak veriyi iki boyuta indirger. Sonrasında, bu iki boyutlu 
veriyi kullanarak veriyi görselleştirir. Kodun adım adım açıklaması aşağıda verilmiştir:

fetch_openml("mnist_784", version=1): MNIST veri setini OpenML'den indirir. 
Bu veri seti, 0'dan 9'a kadar el yazısıyla yazılmış rakamların gri tonlamalı görüntülerinden 
oluşur. Her bir görüntü 28x28 piksel boyutundadır, bu nedenle toplamda 784 özellik 
(piksel değeri) vardır.
X = mnist.data: Veri setindeki görüntülerin piksel değerlerini bir matris olarak alır.
y = mnist.target.astype(int): Etiketleri (yani, hangi rakamın yazılı olduğu bilgisi) 
tam sayıya çevirir. Bu etiketler 0'dan 9'a kadar olan rakamları temsil eder.
"""

mnist = fetch_openml("mnist_784", version=1)

X = mnist.data
y = mnist.target.astype(int)

"""
TSNE(n_components=2): t-SNE algoritması iki boyuta indirgeme yapacak şekilde tanımlanır. 
t-SNE, yüksek boyutlu verileri düşük boyutlara indirgemek için kullanılan, 
özellikle veri kümeleri arasındaki lokal benzerlikleri korumaya odaklanan bir algoritmadır.
X_tsne = tsne.fit_transform(X): t-SNE algoritması MNIST veri setine uygulanır ve veriler 
iki boyuta indirilir. Sonuç olarak, her görüntü iki boyutlu bir vektörle temsil edilir.
"""

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

plt.figure()
plt.scatter(X_tsne[:,0], X_tsne[:,1], c = y, cmap = "tab10", alpha=0.6)
plt.title("TSNE of MNIST Dataset")
plt.xlabel("T-SNE Dimension 1")
plt.ylabel("T-SNE Dimension 2")

"""
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", alpha=0.6): t-SNE ile indirgenmiş 
veriler, iki boyutlu bir grafikte görselleştirilir. Her bir nokta, MNIST veri setindeki bir 
görüntüyü temsil eder. Renkler, görüntülerin temsil ettiği rakamları gösterir.
c=y: Verinin etiketlerine (yani rakamlara) göre renkler ayarlanır.
cmap="tab10": Renk paleti olarak tab10 kullanılır, bu palet 10 farklı renk içerir.
alpha=0.6: Noktaların şeffaflık seviyesi ayarlanır, böylece üst üste binen noktalar daha 
belirgin hale gelir.
plt.title("TSNE of MNIST Dataset"): Grafiğe başlık ekler.
plt.xlabel("T-SNE Dimension 1") ve plt.ylabel("T-SNE Dimension 2"): Eksenlere etiket ekler.
"""

"""
Bu kod, MNIST veri setini t-SNE kullanarak iki boyuta indirger ve ardından bu iki boyutlu 
verileri görselleştirir. Bu görselleştirme, verinin yüksek boyutlu uzayda nasıl dağıldığını 
ve farklı rakamların birbiriyle nasıl ilişkilendiğini görmemize yardımcı olur. 
t-SNE, özellikle karmaşık ve yüksek boyutlu veri kümelerini anlamlı şekilde görselleştirmek 
için sıkça kullanılır.
"""