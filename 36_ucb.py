import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

#Random Selection (Rasgele Seçim)
'''
import random

N = 10000
d = 10 
toplam = 0
secilenler = []
for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
    toplam = toplam + odul
    
    
plt.hist(secilenler)
plt.show()
'''
import math
#UCB
N = 10000 # 10.000 tıklama
d = 10  # toplam 10 ilan var
#Ri(n)
oduller = [0] * d #ilk basta butun ilanların odulu 0
#Ni(n)
tiklamalar = [0] * d #o ana kadarki tıklamalar
toplam = 0 # toplam odul
secilenler = []
for n in range(1,N):
    ad = 0 #seçilen ilan
    max_ucb = 0
    for i in range(0,d):
        if(tiklamalar[i] > 0):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2* math.log(n)/tiklamalar[i])
            ucb = ortalama + delta
        else:
            ucb = N*10
        if max_ucb < ucb: #max'tan büyük bir ucb çıktı
            max_ucb = ucb
            ad = i          
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad]+ 1
    odul = veriler.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
    oduller[ad] = oduller[ad]+ odul
    toplam = toplam + odul
print('Toplam Odul:')   
print(toplam)

plt.hist(secilenler)
plt.show()

"""

Upper Confidence Bound (UCB) algoritması, Çok Kollu Haydut (Multi-Armed Bandit) 
problemini çözmek için kullanılan etkili ve popüler bir yöntemdir. 
UCB, keşif (exploration) ve sömürü (exploitation) arasındaki dengeyi sağlar ve her bir 
seçeneğin (makine veya kol) potansiyelini üst güven sınırını (upper confidence bound) 
kullanarak değerlendirir.

UCB Algoritmasının Temel Prensipleri
UCB algoritmasının temel fikri, her bir kolun (makinenin) potansiyel ödülünün 
bir üst güven sınırını hesaplayarak, bu sınırı maksimize eden kolu seçmektir. 
Bu sayede hem yeterli keşif yapılır hem de bilinen iyi seçenekler sömürülür.

Adım Adım UCB Algoritması

Başlatma: Her bir kolu en az bir kez çekerek başlatılır ve ortalama ödüller hesaplanır.

UCB Hesaplama: Her bir kol için yukarıdaki UCB formülü kullanılarak bir değer hesaplanır.

Seçim: En yüksek UCB değerine sahip kol seçilir.

Güncelleme: Seçilen kol için elde edilen ödül gözlemlenir ve bu kolun ortalama ödülü ve 
çekiş sayısı güncellenir.

Tekrarlama: Adım 2'den itibaren süreç tekrarlanır.


UCB Algoritmasının Kullanım Alanları

UCB algoritması,çok kollu haydut problemlerinin birçok farklı uygulama alanında kullanılır:

Online Reklamcılık: Hangi reklamın daha fazla tıklama alacağını belirlemek için kullanılır.
Öneri Sistemleri: Kullanıcılara sunulan önerilerin etkililiğini artırmak.
Klinik Denemeler: Farklı tedavi yöntemlerinin etkinliğini belirlemek.
A/B Testleri: Farklı versiyonların performansını sürekli olarak optimize etmek.
UCB Algoritmasının Avantajları ve Dezavantajları

Avantajları:

Teorik Dayanak: UCB algoritması, keşif ve sömürme arasında optimal bir denge kurar.
Kolay Uygulanabilirlik: Algoritma, basit ve anlaşılır bir formülle çalışır.
Adaptif: Değişen koşullara hızlı bir şekilde uyum sağlar.

Dezavantajları:

Keşif Parametresi (c): Doğru keşif parametresinin belirlenmesi zor olabilir ve 
problem bağımlıdır.
Hesaplama Maliyeti: Büyük ölçekli problemlerde hesaplama maliyeti artabilir.

UCB Algoritmasının Örneği
Bir online reklamcılık senaryosunda, üç farklı reklam (A, B, C) olduğunu ve 
her bir reklamın tıklanma oranını optimize etmek istediğinizi düşünelim. 
UCB algoritmasını kullanarak en iyi reklamı belirlemek için aşağıdaki adımları izleyebilirsiniz:

Başlatma: Her reklam en az bir kez gösterilir ve tıklama oranları hesaplanır.
UCB Hesaplama: Her reklam için UCB değeri hesaplanır.
Seçim: En yüksek UCB değerine sahip reklam gösterilir.
Güncelleme: Gösterilen reklamın tıklama oranı güncellenir.
Tekrarlama: Süreç tekrar edilir.
Bu şekilde, zamanla UCB algoritması,en yüksek tıklama oranına sahip reklamı daha sık 
göstererek tıklama oranlarını maksimize eder.

UCB algoritması, çok kollu haydut problemlerinin çözümünde etkili ve güçlü bir yöntem 
olup, birçok gerçek dünya uygulamasında başarılı sonuçlar elde etmiştir.

"""