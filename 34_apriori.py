# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('sepet.csv', header = None)

t = []
for i in range (0,7501):
    t.append([str(veriler.values[i,j]) for j in range (0,20)])

from apyori import apriori
kurallar = apriori(t,min_support=0.01, min_confidence=0.2, min_lift = 3, min_length=2)

print(list(kurallar))


"""

Association Rule Mining Association Rule Learning ARM ARL

bunu alanlar bunu da aldı
bunu izleyenler bunu da izledi

bir eylem tekrar ediyorsa ürün marka vs üzerinden bunu yakalamak amacımı
bebek bezi alanlar bira aldı o zaman kampanya yap
amazon netflix bunu alanlar bunu da alır bunu izleyenler bunu da izler

Causation Nedensellik   sıcaklar ile dondurma satışı arasındaki ilişki
Correlation İlişkisellik  dondurma satışı ile köpek balığı saldırısı arasındaki ilişki

ML neden olduğu kısmıyla ilgilenmez aralarında ilişki var mı ona bakar

Support  a yı içerenler / toplam eylem  a nın normalde ne kadar sattığı
Confidence a->b  a ve b yi içerenler / a yı içerenler a nın b ile birlikte ne kadar sattığı
Lift a->b   Confidence a->b / support b  b nin a nın satışını ne kadar etkilediği
lift ne kadar büyükse o kadar pozitif yönde olumlu etkilenir

Apriori verilere bakar her eylemin Frekansını çıkarır min support değeri veririz
bunun altında kalıyorsa elenir min support şu demek pozitif anlamda etkisi en az olan
elenir diğerleriyle devam eder

Nerelerde kullanılır

Complex Event Processing
Kampanya
Davranış Tahmini
Yönlendirilmiş ARM
Zaman serisi Analizi

"""

"""

Association Rule Mining, veri madenciliği alanında sıkça kullanılan bir tekniktir ve 
genellikle büyük veri kümelerindeki öğeler arasındaki ilişkiyi keşfetmek için kullanılır. 
Bu teknik, özellikle perakende ve market sepeti analizlerinde popülerdir. Temel amacı, 
belirli bir veri setinde birlikte sıkça ortaya çıkan öğe gruplarını belirlemektir.

Destek (Support): Bir öğe kümesinin veri setinde ne kadar yaygın olduğunu gösterir. 
Bir öğe kümesinin destek değeri, bu öğe kümesinin ortaya çıktığı işlem sayısının toplam 
işlem sayısına oranıdır.

Güven (Confidence): Bir kuralın doğruluğunu gösterir. Eğer A ve B öğeleri arasında bir 
ilişki varsa, A'nın olduğu işlemlerden kaçında B'nin de bulunduğunu ifade eder.

Kaldıraç (Lift): İki öğenin birbirine olan bağımlılığını ifade eder. 
Birlikte görülme olasılıklarının, bağımsız olarak görülme olasılıklarına oranını gösterir.

Adım Adım İşleyiş:

Veri Toplama: İşlemlerden veya kullanıcı aktivitelerinden oluşan bir veri seti toplanır.
Destek Hesaplama: Her bir öğe kümesi için destek değeri hesaplanır ve belirli bir 
eşik değerin üzerindekiler seçilir.
Kuralların Üretilmesi: Sıkça birlikte görülen öğe kümeleri belirlendikten sonra, 
bu kümeler arasındaki ilişki kuralları çıkarılır.
Güven ve Kaldıraç Hesaplama: Oluşturulan kuralların güven ve kaldıraç değerleri hesaplanır.
Kural Seçimi: Belirlenen eşik değerlerin üzerindeki güven ve kaldıraç değerine sahip kurallar seçilir.


Örnek:
Bir marketin işlem veri tabanında aşağıdaki gibi işlemler olduğunu düşünelim:

{Ekmek, Süt, Peynir}
{Ekmek, Süt}
{Süt, Yumurta}
{Ekmek, Yumurta}
Bu işlemleri kullanarak destek, güven ve kaldıraç değerlerini hesaplayabiliriz. Örneğin, {Ekmek} ve {Süt} öğeleri arasındaki ilişkiyi analiz edelim:

Destek(Ekmek): 3/4 = 0.75
Destek(Süt): 3/4 = 0.75
Destek(Ekmek ve Süt): 2/4 = 0.5
Güven(Ekmek ⇒ Süt): 0.5 / 0.75 = 0.67
Kaldıraç(Ekmek ⇒ Süt): 0.5 / (0.75 * 0.75) = 0.89

Bu hesaplamalar sonucunda, market işlemlerinde ekmek ve sütün birlikte 
sıkça alındığını ve aralarında bir ilişki olduğunu söyleyebiliriz.

Kullanılan Algoritmalar:

Apriori Algoritması: En popüler ve yaygın kullanılan algoritmalardan biridir. 
İlk olarak tek öğeli kümelerin destek değerlerini hesaplar, ardından bunları genişleterek 
büyük öğe kümelerini oluşturur. BFS kullanır
FP-Growth Algoritması: Apriori algoritmasına göre daha verimli olan bu algoritma, 
veri setini sık örüntü ağacı (Frequent Pattern Tree) olarak temsil eder ve 
bu yapıyı kullanarak hızlı bir şekilde sık öğe kümelerini bulur.
Association Rule Mining, özellikle pazarlama stratejileri geliştirmek, 
müşteri sepeti analizleri yapmak ve çapraz satış fırsatlarını belirlemek için 
yaygın olarak kullanılır.

"""


"""

Apriori Algoritması, Association Rule Mining alanında kullanılan temel bir algoritmadır 
ve büyük veri kümelerindeki sık öğe kümelerini ve bu kümeler arasındaki ilişkileri 
keşfetmek için tasarlanmıştır. Bu algoritma, özellikle market sepeti analizlerinde 
sıklıkla kullanılır.

Apriori Algoritmasının Temel Prensipleri

Monotonluk (Apriori) İlkesi:

Eğer bir öğe kümesi sık değilse (yani destek değeri belirli bir eşik değerin altındaysa)
bu öğe kümesinin herhangi bir alt kümesi de sık olamaz.
Bu ilke, arama alanını daraltarak algoritmanın verimliliğini artırır. 
Çünkü, destek değeri düşük olan büyük kümelerin alt kümelerini hesaplamaya gerek kalmaz.

İteratif (Aşamalı) Yapı:

Algoritma,bir öğe kümeleri listesiyle başlar ve her iterasyonda bu kümeleri genişleterek 
bir sonraki öğe kümelerini oluşturur.
İlk olarak tek öğeli kümeler değerlendirilir, ardından iki öğeli, üç öğeli vb. 
kümeler oluşturulur.

Apriori Algoritmasının Adımları

Aday Öğeler Kümesini Bulma (Candidate Generation):

Başlangıçta, tek öğeli aday kümeler oluşturulur ve destek değerleri hesaplanır.
Destek değeri minimum destek eşik değerinden büyük olan kümeler, 
bir sonraki iterasyon için aday olarak seçilir.
Destek Değerlerini Hesaplama:

Her iterasyonda aday kümelerin destek değerleri hesaplanır. Bu destek değerleri, 
veri setindeki bu kümelerin kaç işlemde yer aldığını gösterir.

Adayları Filtreleme:

Hesaplanan destek değerlerine göre, minimum destek eşik değerinin altında kalan 
aday kümeler elenir.

Küme Genişletme (Join Step):

Kalan aday kümeler, bir sonraki iterasyon için daha büyük kümeleri oluşturmak üzere 
birleştirilir (join).

Tekrarlama (Iteration):

Yukarıdaki adımlar, belirlenen bir maksimum öğe kümesi büyüklüğüne ulaşılana veya 
yeni aday küme bulunamayacak duruma gelene kadar tekrarlanır.

Apriori Algoritması Örneği
Diyelim ki elimizde aşağıdaki işlemler (transaction) var:

T1: {Ekmek, Süt}
T2: {Ekmek, Süt, Yumurta}
T3: {Süt, Yumurta}
T4: {Ekmek, Yumurta}
Minimum destek eşik değeri 2 olsun.

1. İterasyon:

Adaylar: {Ekmek}, {Süt}, {Yumurta}
Destek değerleri:
Destek({Ekmek}) = 3
Destek({Süt}) = 3
Destek({Yumurta}) = 3

2. İterasyon:

Adaylar: {Ekmek, Süt}, {Ekmek, Yumurta}, {Süt, Yumurta}
Destek değerleri:
Destek({Ekmek, Süt}) = 2
Destek({Ekmek, Yumurta}) = 2
Destek({Süt, Yumurta}) = 2

3. İterasyon:

Aday: {Ekmek, Süt, Yumurta}
Destek değeri:
Destek({Ekmek, Süt, Yumurta})= 1 (minimum destek eşik değerinden küçük olduğu için elenir)
Bu örnekte, minimum destek değeri 2 olan tüm sık öğe kümeleri 
{Ekmek}, {Süt}, {Yumurta}, {Ekmek, Süt}, {Ekmek, Yumurta}, {Süt, Yumurta} olarak bulunur.

Apriori Algoritmasının Avantajları ve Dezavantajları

Avantajları:

Basit ve anlaşılması kolay bir algoritmadır.
Küçük ve orta ölçekli veri setlerinde etkili çalışır.

Dezavantajları:

Büyük veri kümelerinde ve düşük minimum destek eşik değerlerinde verimliliği düşer.
Hesaplama maliyeti yüksektir çünkü her iterasyonda tüm aday kümelerin destek değerlerini 
hesaplamak gereklidir.
Apriori algoritması,veri madenciliği ve özellikle alışveriş sepeti analizi gibi alanlarda 
yaygın olarak kullanılmakta ve sıklıkla diğer gelişmiş algoritmaların 
temelini oluşturmaktadır.

"""



"""
apriori gibi bütün veri setine bakması gerekmez 1 ve 3 ün ortak şeyini bulmak için 13

2 4 5 e gerek yoktur  1 ve 2 yi bilerek direkt olarak 12 ye bakabilir ortak olanlar kalır

135 235 vs
"""



"""

Eclat Algoritması, Association Rule Mining alanında kullanılan bir başka önemli 
algoritmadır. Apriori algoritmasına benzer şekilde sık öğe kümelerini bulmak için 
kullanılır, ancak Eclat algoritması farklı bir yaklaşım benimser. 
Eclat, "Equivalence Class Clustering and bottom-up Lattice Traversal" anlamına gelir ve 
dikey veri düzenlemesi kullanarak çalışır. DFS İLE ÇALIŞIR

Eclat Algoritmasının Temel Prensipleri
Dikey Veri Düzenlemesi:

Eclat algoritması, veri kümesini yatay yerine dikey olarak temsil eder. 
Bu, her öğenin hangi işlemlerde yer aldığını gösteren bir liste oluşturmak anlamına gelir.
Örneğin, Ekmek öğesi {T1, T2, T4} işlemlerinde yer alıyorsa, 
Ekmek için dikey veri temsilcisi {T1, T2, T4} olacaktır.

Kesişim Operasyonu:

Algoritma, sık öğe kümelerini bulmak için dikey veri temsilcilerini 
(öğe kümelerinin işlem listelerini) kesiştirir. Bu sayede, daha büyük öğe kümelerinin 
destek değerleri hesaplanabilir.
İki öğe kümesinin kesişimi, bu iki kümenin birlikte yer aldığı işlemleri verir.

Eclat Algoritmasının Adımları
Veri Dönüşümü:

İlk olarak, veri seti dikey bir forma dönüştürülür. 
Bu, her öğenin bulunduğu işlem listelerinin oluşturulması anlamına gelir.
Kesişim ve Destek Hesaplama:

Algoritma, her iterasyonda öğe kümelerinin işlem listelerini kesiştirerek 
daha büyük kümeler oluşturur ve bu kümelerin destek değerlerini hesaplar.
Kesişim işlemi, iki öğe kümesinin kesişim listesinin boyutunu, 
yani destek değerini verir.

Öğe Kümesi Genişletme:

Sık öğe kümeleri bulunana kadar işlem listeleri kesiştirilerek daha büyük kümeler oluşturulur.
Kesişim sonucunda destek değeri minimum destek eşik değerinden büyük olan kümeler 
genişletilmeye devam eder.


Eclat Algoritmasının Örneği
Diyelim ki elimizde aşağıdaki işlemler var:

T1: {Ekmek, Süt}
T2: {Ekmek, Süt, Yumurta}
T3: {Süt, Yumurta}
T4: {Ekmek, Yumurta}
Minimum destek eşik değeri 2 olsun.

1. Dikey Veri Temsilcileri:

Ekmek: {T1, T2, T4}
Süt: {T1, T2, T3}
Yumurta: {T2, T3, T4}
2. Kesişim ve Destek Hesaplama:

Ekmek ∩ Süt: {T1, T2} (destek = 2)
Ekmek ∩ Yumurta: {T2, T4} (destek = 2)
Süt ∩ Yumurta: {T2, T3} (destek = 2)
3. Daha Büyük Küme Kesişimleri:

Ekmek ∩ Süt ∩ Yumurta: {T2} (destek = 1, elenir)
Bu örnekte, minimum destek değeri 2 olan sık öğe kümeleri {Ekmek}, {Süt}, {Yumurta}, 
{Ekmek, Süt}, {Ekmek, Yumurta}, {Süt, Yumurta} olarak bulunur.

Eclat Algoritmasının Avantajları ve Dezavantajları

Avantajları:

Dikey veri temsilcisi kullanımı sayesinde bazı durumlarda Apriori'den daha verimli olabilir.
Kesişim işlemleri ile daha hızlı hesaplamalar yapılabilir.

Dezavantajları:

Büyük veri kümelerinde ve çok sayıda öğe içeren veri setlerinde performansı düşebilir.
Bellek kullanımı yüksek olabilir çünkü dikey veri temsilcileri saklanmalıdır.
Eclat algoritması, özellikle veri setindeki işlemlerin az sayıda olduğu ve 
öğe sayısının fazla olduğu durumlarda etkili bir yöntemdir. 
Dikey veri düzenlemesi ve kesişim tabanlı hesaplamalar sayesinde sık öğe kümelerinin 
bulunmasında verimli bir yaklaşım sunar.

"""