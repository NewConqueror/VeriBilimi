import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

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


"""

Random Selection (Rastgele Seçim), Çok Kollu Haydut (Multi-Armed Bandit) problemini 
çözmek için kullanılan en basit yöntemlerden biridir. 
Bu yöntem, keşif (exploration) ve sömürü (exploitation) arasında denge kurma ihtiyacını 
göz ardı ederek, her bir kolu (seçeneği) eşit olasılıkla rastgele seçer. 
Yani, her adımda herhangi bir kolun seçilme şansı aynıdır.

Random Selection Algoritmasının Prensipleri
Random Selection algoritması, her adımda bağımsız ve rastgele olarak bir kol seçer. 
Bu yöntem, genellikle daha karmaşık stratejilerin performansını değerlendirmek veya 
bir başlangıç noktası olarak kullanılır.

Adım Adım Random Selection Algoritması
Başlatma: Herhangi bir başlangıç ayarı gerektirmez.
Seçim: Her adımda, mevcut kollar arasından rastgele bir kol seçilir.
Güncelleme: Seçilen koldan elde edilen ödül gözlemlenir, ancak bu bilgi gelecekteki 
seçimler için doğrudan kullanılmaz.
Tekrarlama: Süreç belirlenen zaman veya çekiş sayısı tamamlanana kadar tekrarlanır.
Random Selection Algoritmasının Avantajları ve Dezavantajları
Avantajları:

Basitlik: Uygulaması ve anlaşılması son derece kolaydır.
Hız: Hesaplama maliyeti çok düşüktür, çünkü herhangi bir karmaşık hesaplama yapmayı 
gerektirmez.
Keşif: Tüm kolları eşit olasılıkla seçtiği için geniş bir keşif yapılır.

Dezavantajları:

Etkisizlik: Performansı optimize edemez, çünkü gelecekteki seçimler için ödülleri 
kullanarak öğrenme yapmaz.
Optimal Olmama: Optimal kolu bulmak ve onu sömürmek için herhangi bir mekanizması yoktur.
Uzun Vadeli Başarı Eksikliği:Rastgele seçim, uzun vadede en yüksek toplam ödülü elde edemez.


Random Selection'in Kullanım Alanları
Karşılaştırma ve Benchmarking: Diğer daha karmaşık algoritmaların performansını 
değerlendirmek için bir temel olarak kullanılabilir.
Basit Sistemler: Keşif ve sömürme dengesi gerektirmeyen basit sistemlerde kullanılabilir.
Başlangıç Aşaması: Daha karmaşık algoritmaların başlangıçta yeterli veri toplayabilmesi 
için kullanılabilir.

Random Selection Örneği
Bir online reklamcılık senaryosunda, üç farklı reklam (A, B, C) olduğunu ve her bir 
reklamın tıklanma oranını optimize etmek istediğinizi düşünelim. 
Random Selection algoritmasını kullanarak reklam seçimi şu şekilde yapılır:

Başlatma: İlk adımda herhangi bir ön bilgi olmadan başlatılır.
Seçim: Rastgele bir reklam seçilir ve gösterilir (örneğin, A, B veya C).
Güncelleme: Seçilen reklamın tıklanma oranı gözlemlenir, ancak bu bilgi gelecekteki 
seçimleri etkilemez.
Tekrarlama: Her adımda bu süreç rastgele bir seçimle tekrarlanır.
Bu şekilde, zamanla her reklam yaklaşık olarak eşit sayıda gösterilmiş olur, 
ancak tıklanma oranlarını optimize etmek için daha karmaşık bir algoritma kullanılması 
gerekir.

Karşılaştırma: Random Selection ve UCB
Random Selection: Her adımda rastgele bir seçim yapar ve ödül bilgilerini gelecekteki 
seçimlerde kullanmaz. Keşif ve sömürme arasında denge kurmaz.
UCB (Upper Confidence Bound): Her adımda en yüksek UCB değerine sahip kolu seçer ve 
geçmiş ödül bilgilerini kullanarak keşif ve sömürme arasında denge kurar. 
Uzun vadede en yüksek ödülü maksimize etmeye çalışır.
Özetle, Random Selection basit ve hızlı bir yöntem olup, daha karmaşık ve 
optimize edilmiş yöntemlerin performansını değerlendirmek veya başlangıç aşamasında ,
veri toplamak için kullanışlıdır. Ancak, optimal sonuçlar elde etmek için genellikle 
daha gelişmiş algoritmalar tercih edilir.

"""







