"""
Unsupervised gözetimsiz öğrenme

modele herhangi bir ön kabul sınıf vermiyoruz bütün hepsini aynı görüyor
sonra kendisi sınıflar oluşturuyor yakınlığa göre vs

normalde biz sınıf verirdik erkekler kadınlar a lar b ler vs verdiğimiz sınıflar arasında 
ayrım yapmaya çalışırdı bir çizgi ile

ama kümeleme bölütleme Clustering de böyle bir şey yok hepsini aynı görür
sonra kendisi erkek kadın a b diye ayrım yapar farklı algoritmalar ile

Müşteri segmentasyonu
    Collaboration Filtering
    özel kampanyalar
    Tehdit ve Sahtekarlık yakalama
    Eksik verilerin tamamlanması
    Verinin alt kümesi üzerinde yapılan bütün işlemler

ör Müşterileri 3 e böler zengin orta fakir sınıf diye bunlara özel ürün önerilerinde bulunur
zengin e saat orta ya teknoloji fakir e gıda ürünleri gösterilmesi gibi
senin aldığın bir ürünü senin sınıfındaki diğer kişilerin de alması yüksektir 

Pazar segmentasyonu
    Davranışsal segmentasyon
    Demografik segmentasyon
    Psikolojik segmentasyon
    Coğrafi segmentasyon

Sağlık ve görüntü işleme
    kanser tespiti vs
    nesne tespiti




Clustering, veri madenciliği ve makine öğrenmesi alanlarında kullanılan bir tekniktir. 
Amacı, veri setindeki benzer veri noktalarını gruplandırarak anlamlı kümeler oluşturmaktır. Bu süreç, verileri belirli özelliklere göre sınıflandırarak analiz yapmayı ve örüntüler keşfetmeyi sağlar. İşte clustering hakkında bazı temel bilgiler:

Clustering Teknikleri
K-Means Clustering: Verileri k adet küme merkezine yakınlıklarına göre gruplar. 
Her veri noktası en yakın merkezine atanır ve merkezler iteratif olarak güncellenir.
Hierarchical Clustering: Verileri hiyerarşik bir yapıda gruplar.
Bu, aglomeratif (alt seviyelerden başlayarak birleşen) veya 
divizif (üst seviyeden başlayarak ayıran) yöntemlerle yapılabilir.
DBSCAN (Density-Based Spatial Clustering of Applications with Noise): 
Yoğunluk bazlı bir yöntemdir. Yüksek yoğunluklu bölgelerdeki veri noktalarını 
kümeler ve gürültü noktalarını belirler.
Gaussian Mixture Models (GMM): Verilerin, normal dağılımlarla modellenebileceği varsayımına 
dayanır. Veri noktaları, belirli bir olasılıkla her bir kümenin parçası olarak 
sınıflandırılır.
Clustering Uygulamaları
Müşteri Segmentasyonu: Pazarlamada, müşterileri benzer satın alma davranışlarına 
göre gruplandırmak.
Görüntü İşleme: Benzer özelliklere sahip pikselleri gruplandırarak görüntü segmentasyonu 
yapmak.
Biyoinformatik: Gen ekspresyon verilerini analiz ederek benzer genleri veya hastalıkları 
gruplandırmak.
Sosyal Ağ Analizi: Sosyal ağlarda benzer ilgi alanlarına veya etkileşimlere sahip 
kullanıcıları belirlemek.

"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans

kmeans = KMeans ( n_clusters = 3, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
sonuclar = []
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11),sonuclar)

"""
Nasıl çalışır

Kaç küme olacağı kullanıcdan paremetre olarak alınır n cluster denir
Rastgele k tane merkez noktası seçilir
Her veri örneği en yakın merkez noktasına göre ilgili kümeye atanır
Her küme için yeni merkez noktaları hesaplanarak merkez noktaları kaydırılır
yeni merkez noktalarına göre tekrarlanır stabil hale gelene kadar

K-Means Başlangıç Noktası Tuzağı

merkez noktalarını rastgele seçtiğimiz için bazen olması gerekenden ideal olandan 
farklı olarak sonuçlar alabiliriz tek küme olması gereken 2 kümeye bölünebilir
2 küme olması gereken tek küme kalabilir vs

bunu K-Means++ ile halledebiliriz

Rastgele seçilen noktalardan merkeze olan uzaklık hesaplanır Dx
bunun karesi olasılık olarak alınır Dx^2 yani yakın olan 1 zar atar uzak olan 4 zar atar
uzak olana doğru yaklaşır 
sonra bu işlem tekrar edilir vs

Cluster lar kendi bulunduğu kümeden başka bir kümeye atlama yapabilir kmeans++ ile

WCSS değeri bize yardım eder
her noktanın merkeze göre uzaklığının karesi alınır ve toplanır bu da WCSS değerini verir

1 Cluster varken hesaplanır 2 Cluster varken hesaplanır 3 Cluster varken hesaplanır

Grafiğe döktüğün zaman ani bir düşüş sonra yavaşça 0 a inen bir değer görürsün
Cluster sayısı çok olursa model öğrenmez ezberler
bunun için dirsek noktası dediğimiz ani düşüşten sonraki noktaları kullanırız
WCSS değerinin ciddi miktarda düştüğü yer yani

çok CLuster ezberleme demek 

"""



"""

K-Means algoritması, popüler ve yaygın olarak kullanılan bir clustering (kümeleme) 
yöntemidir. Amacı, veri setini k adet küme olacak şekilde bölmektir. 
Her küme, veri noktalarının yakın olduğu bir merkeze sahiptir. 
İşte K-Means algoritmasının nasıl çalıştığını ve temel kavramlarını açıklayan bir özet:

K-Means Algoritmasının Temel Adımları
Başlangıç Küme Merkezlerini Belirleme:

Rastgele olarak veya belirli bir yönteme göre k adet başlangıç küme merkezi seçilir.

Veri Noktalarını Küme Merkezlerine Atama:

Her veri noktası, en yakın küme merkezine atanır. Bu işlem, genellikle Öklidyen mesafeye 
göre yapılır.

Küme Merkezlerini Güncelleme:

Her küme için yeni merkez, kümedeki tüm veri noktalarının ortalaması alınarak hesaplanır.

Adım 2 ve 3'ü Tekrarlama:

Küme merkezleri değişene veya belirli bir durma kriteri sağlanana kadar 
(örneğin, küme merkezleri sabit kalana kadar) adımlar tekrarlanır.

K-Means Algoritmasının Özellikleri
Deterministik Olmaması: Rastgele başlangıç merkezlerine bağlı olarak farklı sonuçlar 
elde edilebilir. Bu nedenle, algoritma genellikle birden çok kez çalıştırılarak 
en iyi sonuç seçilir.
Küme Sayısı (k): Kullanıcının önceden belirlemesi gerekir. Yanlış k seçimi, kötü sonuçlara 
yol açabilir.
Ölçeklenebilirlik: Büyük veri setleri üzerinde hızlı çalışabilir, 
ancak büyük veri kümeleri için başlangıç merkezlerinin dikkatli seçilmesi önemlidir.

Avantajları
Basit ve Anlaşılır: Uygulaması ve anlaması kolaydır.
Hızlı ve Verimli: Büyük veri setlerinde bile hızlı bir şekilde çalışabilir.

Dezavantajları
Küme Sayısının Önceden Belirlenmesi: k değerinin önceden bilinmesi zor olabilir.
Duyarlılık: Başlangıç noktalarına ve veri setindeki gürültüye karşı duyarlıdır.
Küme Şekli: Küme şekillerinin küresel ve benzer boyutlarda olması varsayımına dayanır.

"""

"""

K-Means başlangıç tuzağı (initialization trap), K-Means algoritmasının rastgele başlangıç 
noktalarına duyarlı olmasından kaynaklanan bir problemdir. 
Bu, algoritmanın rastgele seçilen ilk küme merkezlerine bağlı olarak farklı sonuçlar 
vermesine neden olabilir. Başlangıç küme merkezlerinin kötü seçilmesi, 
algoritmanın yerel minimumlarda sıkışmasına ve dolayısıyla kötü sonuçlara yol açabilir.
İşte başlangıç tuzağının detayları ve bu tuzaktan kaçınma yolları:

Başlangıç Tuzağı Problemi
Rastgele Başlangıç: K-Means algoritması genellikle rastgele başlangıç küme merkezleri seçer. 
Eğer bu merkezler iyi seçilmezse, algoritma optimal olmayan bir çözümde sıkışabilir.
Yerel Minimum: K-Means algoritması, iteratif olarak küme merkezlerini günceller ve 
her iterasyonda toplam hata (veri noktalarının küme merkezlerine olan toplam mesafesi) 
azaltılmaya çalışılır. Ancak, rastgele başlangıç noktaları algoritmanın küresel minimum 
yerine yerel minimuma ulaşmasına neden olabilir.

Başlangıç Tuzağından Kaçınma Yöntemleri
Birden Çok Çalıştırma (Multiple Runs): Algoritmayı farklı rastgele başlangıç noktalarıyla 
birden çok kez çalıştırmak ve en iyi sonuçları seçmek. Bu, başlangıç noktalarının 
kötü seçilme olasılığını azaltır.


K-Means++ Başlangıç (K-Means++ Initialization): K-Means++ algoritması, 
başlangıç merkezlerini seçerken daha akıllı bir yöntem kullanır. 
İlk merkezi rastgele seçer, sonraki merkezleri ise mevcut merkezlere olan uzaklıkları 
göz önünde bulundurarak belirler. Bu yöntem, başlangıç merkezlerinin daha iyi seçilmesini 
ve yerel minimum tuzağından kaçınmayı sağlar.


K-Means++ Yöntemi
K-Means++ başlangıç yöntemi, başlangıç merkezlerinin seçilme sürecini 
daha stratejik hale getirir ve algoritmanın daha iyi sonuçlar üretmesini sağlar. 
İşte K-Means++ algoritmasının adımları:

İlk küme merkezi rastgele seçilir.
Her veri noktası için mevcut küme merkezlerine olan en kısa mesafe hesaplanır.
Yeni bir küme merkezi,bu mesafelerin karesi ile orantılı bir olasılık dağılımı kullanılarak 
seçilir. Uzak olan noktalara merkez olma olasılığı daha yüksektir.
Adım 2 ve 3, 
k merkez seçilene kadar tekrarlanır.
Bu yöntem, başlangıç noktalarının daha dengeli ve veri setindeki dağılımı daha iyi 
temsil etmesini sağlar.

Özet
K-Means başlangıç tuzağı, rastgele seçilen başlangıç küme merkezlerinin algoritmanın 
sonuçlarını olumsuz etkileyebilmesi durumudur. Bu tuzaktan kaçınmak için K-Means++ 
başlangıç yöntemi veya algoritmayı birden çok kez çalıştırarak en iyi sonucu seçmek gibi 
stratejiler kullanılabilir. Bu yöntemler, K-Means algoritmasının daha güvenilir ve tutarlı 
sonuçlar üretmesine yardımcı olur.

"""

"""
WCSS (Within-Cluster Sum of Squares), K-Means algoritmasının performansını değerlendirmek 
ve optimal küme sayısını belirlemek için kullanılan bir metriktir. 
WCSS, her bir kümenin içindeki veri noktalarının küme merkezine olan uzaklıklarının 
karelerinin toplamıdır.Bu değer,kümelerin içindeki veri noktalarının ne kadar yakın olduğunu
ve kümelerin ne kadar kompakt olduğunu gösterir.

WCSS'nin Kullanım Alanları
Optimum Küme Sayısını Belirlemek: K-Means algoritması için en uygun küme sayısını belirlemek 
amacıyla kullanılır. Bu amaçla, "Dirsek Yöntemi" (Elbow Method) adı verilen bir yöntem 
kullanılır.

Dirsek Yöntemi
Dirsek yöntemi, WCSS değerinin küme sayısına bağlı olarak nasıl değiştiğini gözlemleyerek 
optimal küme sayısını belirler. Bu yöntem, genellikle WCSS'nin önemli bir şekilde azalmayı 
bıraktığı ve eğrinin bir "dirsek" (elbow) yaptığı noktayı bulmayı amaçlar.


Özet
WCSS, K-Means algoritmasının küme performansını değerlendirmek ve optimal küme sayısını 
belirlemek için kullanılan bir metriktir. Dirsek yöntemi ile birlikte kullanılarak en uygun 
küme sayısının seçilmesine yardımcı olur. Bu, K-Means algoritmasının doğru bir şekilde 
uygulanmasını ve verimli sonuçlar elde edilmesini sağlar.

"""