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
import random
#UCB
N = 10000 # 10.000 tıklama
d = 10  # toplam 10 ilan var
#Ri(
#Ni(n)
toplam = 0 # toplam odul
secilenler = []
birler = [0] * d
sifirlar = [0] * d
for n in range(1,N):
    ad = 0 #seçilen ilan
    max_th = 0
    for i in range(0,d):
        rasbeta = random.betavariate ( birler[i] + 1 , sifirlar[i] +1)
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
    secilenler.append(ad)
    odul = veriler.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
    if odul == 1:
        birler[ad] = birler[ad]+1
    else :
        sifirlar[ad] = sifirlar[ad] + 1
    toplam = toplam + odul
print('Toplam Odul:')   
print(toplam)

plt.hist(secilenler)
plt.show()

"""

Adım1:
Her aksiyon için aşağıdaki iki sayıyı hesaplayınız 

Ni1(n): o ana kadar ödül olarak 1 gelmesi sayısı
Ni0(n): o ana kadar ödül olarak 0 gelmesi sayısı

Adım2:
Her ilan için aşağıda verilen beta dağılımında bir rastgele sayı üretiyoruz

tetai(n) = Beta(Ni1(n) +1, Ni0(n) + 1 )

Adım3:
En yüksek beta değerine sahip olanı seçiyoruz

"""



"""

### Thompson Sampling

Thompson Sampling, Çok Kollu Haydut (Multi-Armed Bandit) problemini çözmek için 
kullanılan bir başka popüler ve etkili algoritmadır. Bu algoritma, Bayesian olasılık 
teorisini kullanarak her bir kolun (seçeneğin) ödül oranını tahmin eder ve 
en yüksek tahmini ödüle sahip kolu seçer. Thompson Sampling, keşif (exploration) ve 
sömürü (exploitation) arasında doğal bir denge sağlar.

### Temel Prensipler

Thompson Sampling, her bir kolun ödül oranını bir olasılık dağılımı olarak modelleyerek 
çalışır. Bu dağılımlar, gözlemlenen verilere dayalı olarak sürekli güncellenir. 
Algoritma, her adımda bu dağılımlardan örnekler alır ve en yüksek örnek değeri veren 
kolu seçer.

#### Adım Adım Thompson Sampling Algoritması

1. **Başlatma:** Her bir kol için ödül oranlarını temsil eden bir öncelik dağılımı 
(prior distribution) belirlenir. Genellikle Beta dağılımı kullanılır çünkü ödül oranları 
[0, 1] aralığındadır ve Beta dağılımı bu aralığı iyi modeller.

2. **Örnekleme:** Her bir kol için mevcut dağılımdan bir örnek alınır.

3. **Seçim:** En yüksek örnek değerine sahip kol seçilir ve çekilir.

4. **Güncelleme:** Seçilen kolun ödülü gözlemlenir ve bu bilgi kullanılarak 
ilgili kolun dağılımı güncellenir.

5. **Tekrarlama:** Adım 2'den itibaren süreç tekrar edilir.

#### Beta Dağılımı ile Thompson Sampling

Beta dağılımı, iki parametreye sahiptir: alfa (\(\alpha\)) ve beta (\(\beta\)). 
Bu parametreler, başarı ve başarısızlık sayısını temsil eder ve sürekli olarak 
güncellenir.

- Başlangıçta, tüm kollar için \(\alpha\) ve \(\beta\) genellikle 1 olarak başlatılır.
- Her başarı gözlemlendiğinde (ödül alındığında), \(\alpha\) 1 artırılır.
- Her başarısızlık gözlemlendiğinde (ödül alınmadığında), \(\beta\) 1 artırılır.

### Thompson Sampling Örneği

Bir online reklamcılık senaryosunda, üç farklı reklam (A, B, C) olduğunu ve her bir 
reklamın tıklanma oranını optimize etmek istediğinizi düşünelim. 
Thompson Sampling algoritmasını kullanarak reklam seçimi şu şekilde yapılır:

1. **Başlatma:** Her reklam için \(\alpha\) ve \(\beta\) değerlerini 1 olarak başlatın.

2. **Örnekleme:** Her reklam için Beta dağılımından bir örnek alın. Örneğin:
   - Reklam A: Beta(\(\alpha_A, \beta_A\))
   - Reklam B: Beta(\(\alpha_B, \beta_B\))
   - Reklam C: Beta(\(\alpha_C, \beta_C\))

3. **Seçim:** En yüksek örnek değerine sahip reklamı seçin ve gösterin.

4. **Güncelleme:** Seçilen reklamın tıklanma sonucuna göre \(\alpha\) veya \(\beta\) 
değerini güncelleyin.
   - Tıklama: \(\alpha\) 1 artırılır.
   - Tıklama yok: \(\beta\) 1 artırılır.

5. **Tekrarlama:** Süreç tekrar edilir.

### Thompson Sampling'in Avantajları ve Dezavantajları

**Avantajları:**
- **Etkin Keşif ve Sömürü Dengesi:** Olasılık dağılımlarını kullanarak keşif ve sömürme 
arasında doğal bir denge kurar.
- **Adaptif:** Değişen ödül oranlarına hızlı bir şekilde uyum sağlar.
- **Pratik:** Teorik olarak sağlam ve pratikte iyi performans gösterir.

**Dezavantajları:**
- **Hesaplama Maliyeti:** Örnekleme ve güncelleme süreçleri, özellikle büyük ölçekli 
problemlerde hesaplama maliyetini artırabilir.
- **Model Varsayımları:** Dağılımın doğru seçilmesi ve parametrizasyonu önemlidir; 
yanlış varsayımlar performansı olumsuz etkileyebilir.

### Karşılaştırma: Thompson Sampling ve UCB

- **Thompson Sampling:** Bayesian güncellemelerle her kol için olasılık dağılımlarını 
kullanarak seçim yapar. Örnekleme ile keşif ve sömürme arasında denge sağlar.
- **UCB (Upper Confidence Bound):** Her kol için bir güven aralığı hesaplar ve 4
bu aralığın üst sınırını maksimize eden kolu seçer. Teorik olarak sağlamdır ve 
keşif-sömürü dengesini matematiksel olarak kurar.

Her iki algoritma da Çok Kollu Haydut problemini çözmek için güçlü yöntemlerdir ve 
spesifik uygulama ve koşullara bağlı olarak biri diğerine tercih edilebilir. 
Thompson Sampling, adaptif ve esnek doğasıyla birçok gerçek dünya uygulamasında 
etkili bir şekilde kullanılmaktadır.


"""


