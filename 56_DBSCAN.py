from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

"""

Bu kod, DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algoritmasını 
kullanarak örnek bir veri kümesi üzerinde kümeleme işlemi gerçekleştirir ve sonuçları 
görselleştirir.

make_circles: İki iç içe geçmiş dairesel veri kümesi oluşturmak için kullanılır.
DBSCAN: Yoğunluk tabanlı bir kümeleme algoritmasıdır. Gürültü ve rastgele veri noktalarını ,
dikkate alarak, yoğunluk bölgelerine dayalı olarak kümeler oluşturur.
"""

"""
make_circles: 1000 örnekten oluşan, iki iç içe geçmiş daireden oluşan bir 
veri kümesi oluşturur.
factor=0.5: İçteki dairenin yarıçapını dıştaki daireye göre belirler.
noise=0.08: Verilere küçük miktarda gürültü ekler.
random_state=42: Sonuçların tekrarlanabilir olmasını sağlar.
plt.scatter(X[:, 0], X[:, 1]): Üretilen veri kümesini 2 boyutlu bir grafikte noktalar 
halinde gösterir. Bu aşamada, verilerin ham hali görselleştirilir.

eps=0.15: Bir noktanın başka bir noktaya olan maksimum mesafesi. 
Eğer bir noktanın etrafında bu mesafe içinde yeterli sayıda başka nokta varsa, 
o nokta bir küme merkezinin parçası olarak kabul edilir.
min_samples=15: Bir noktanın merkez noktası olarak kabul edilebilmesi için çevresinde 
en az bulunması gereken nokta sayısı.
"""
X, _ = make_circles(n_samples=1000, factor = 0.5, noise = 0.08, random_state=42)
plt.figure()
plt.scatter(X[:, 0], X[:, 1])

dbscan = DBSCAN(eps = 0.15, min_samples = 15)
cluster_labels = dbscan.fit_predict(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = cluster_labels, cmap = "viridis")
plt.title("DBSCAN sonuclari")

"""
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap="viridis"): Her bir veri noktasını, 
DBSCAN algoritması tarafından belirlenen küme etiketine göre renklendirir. 
Farklı kümeler farklı renklerde gösterilir, ve gürültü noktaları (eğer varsa) genellikle -1 
etiketiyle gösterilir.

Bu kod, DBSCAN algoritmasını kullanarak yoğunluk temelli bir kümeleme işlemi gerçekleştirir. 
make_circles fonksiyonu ile oluşturulan iç içe geçmiş dairesel veri kümesinde, 
DBSCAN, kümeleri yoğunluk bölgelerine göre ayırır ve sonuçları görselleştirir. 
DBSCAN, geleneksel k-ortalama gibi kümeleme yöntemlerinden farklı olarak, 
düzensiz şekilli kümeleri ve gürültü noktalarını tanımlamakta daha etkilidir.
"""