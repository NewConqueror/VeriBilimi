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
Reinforced Learning Pekiştirmeli Takviyeli öğrenme 

Makinenin kendi kendine öğrenmesidir doğru bir hareket yaptığında ödül veririz
yanlış bir hareket yaptığında ceza veririz süreç ilerledikçe makine ödül verilen 
davranışı daha çok yapmaya başlar kendi kendine öğrenmiş olur 
en başta biz bir şey tanımlamayız zamanla kendi kendine ayakta durmayı
go oynamayı vs öğrenir Alpha Go

A/B testi kullanıcıya 2 reklam gösterilir A ve B hangisine daha çok tıklandıysa
makine der hee bunun ödülü daha fazla bunu kullanıcıya göstereyim

One armed bandit tek kollu haydut

her makinenin bir dağılımı vardır gözlem ile kazanma şansımız en yüksek olan dağılımı
seçeriz

UCB upper Confidence Bound Üst güven aralığı

Her olayın arkasında bir dağılım vardır

Kullanıcı her seferinde bir eylem yapar

bu eylem karşılığında bir skor döner web tıklaması 1 tıklanmaması 0

Amaç tıklamaları Maksimum a çıkarmaktır

Ajanımız bir aksiyon yapar çevreden geri dönüşler alır gözlem yapar ödül ya da ceza alır
ve kendini iyileştirir

En yüksek UCB yani üst güven aralığına sahip olanı seçeriz HER ZAMAN

Random selection en aptalca algoritmadır temeldir hiçbir ödül ceza sistemi yok
makine rastgele sayı seçiyor ve bir ödül kazanıyor

yazdığın algoritma Rastgele seçen algoritmadan daha fazla ödül kazanmak zorunda
kazanamazsa kötü bir algoritma yazdın demek çünkü rastgele seçim en aptalca olanı

UCB de
Adım1:
her turda(tur sayısı n) her reklam alternatifi (i) için aşağıdaki sayılar tutulur

Ni(n): i sayılı reklamın o ana kadarki tıklama sayısı
Ri(n): o ana kadar ki i reklamından gelen toplam ödül

Adım2:
Yukarıdaki bu iki sayıdan aşağıdaki değerler hesaplanır

O ana kadarki her reklamın ortalama ödülü Ri(n) / Ni(n)

Güven aralığı için aşağı ve yukarı oynama potansiyeli di(n)karekök 3logn/ 2Ni(n)

Adım3: En yüksek UCB değerine sahip olanı alırız

"""

"""

Reinforcement Learning (RL), makine öğrenmesi ve yapay zeka alanında önemli bir dal olup, 
bir ajanın (agent) bir ortam (environment) içinde eylemler gerçekleştirerek ödüller 
(rewards) topladığı ve bu ödülleri maksimize etmeye çalıştığı bir öğrenme yöntemidir. 
RL, özellikle kontrol problemleri, oyunlar ve robotik gibi alanlarda kullanılır.

Temel Kavramlar

Ajan (Agent): Öğrenen ve eylemleri gerçekleştiren varlık. 
Örneğin, bir oyun oynayan yapay zeka.

Ortam (Environment): Ajanın etkileşimde bulunduğu ve eylemlerinin 
sonuçlarını gözlemlediği çevre. Örneğin, bir oyun tahtası veya gerçek dünya.

Durum (State): Ajanın o anki çevresel durumu. Her eylem sonrası ortamın durumu 
değişebilir. Örneğin, bir oyunda ajanın bulunduğu konum ve çevresel faktörler.

Eylem (Action): Ajanın belirli bir durumda gerçekleştirdiği hareket. 
Örneğin, bir robotun ilerlemesi veya bir oyun karakterinin zıplaması.

Ödül (Reward): Ajanın belirli bir eylemi gerçekleştirdikten sonra aldığı geri bildirim. 
Ödül, ajanın doğru veya yanlış eylemleri öğrenmesini sağlar. 
Örneğin, doğru bir hamle için pozitif bir puan, yanlış bir hamle için negatif bir puan.

Politika (Policy): Ajanın hangi durumda hangi eylemi gerçekleştireceğini belirleyen 
strateji. Politika, durumlardan eylemlere bir haritalamadır.

Değer Fonksiyonu (Value Function): Belirli bir durumda olmak veya belirli bir eylemi 
gerçekleştirmek için beklenen toplam ödülü ölçen fonksiyon. Bu fonksiyon, 
uzun vadeli ödülleri maksimize etmeye yardımcı olur.

Reinforcement Learning Algoritmaları
Q-Learning:
SARSA (State-Action-Reward-State-Action):
Policy Gradient Methods: 
Actor-Critic Methods: 

Reinforcement Learning Uygulamaları
Oyunlar: Satranç, Go, ve video oyunları gibi oyunlarda RL, insan seviyesinde veya 
insan üstü performans sergileyen ajanlar geliştirmek için kullanılır. 
Örneğin, DeepMind'in AlphaGo ve AlphaZero projeleri.

Robotik: RL, robotların çeşitli görevleri öğrenmesi için kullanılır. Örneğin, 
robotların nesneleri tutma, yürüme veya otonom araçların hareketlerini öğrenmesi.

Finans: RL, ticaret stratejileri geliştirmek, portföy optimizasyonu ve risk yönetimi gibi 
finansal uygulamalarda kullanılır.

Kontrol Sistemleri:Otonom sistemlerin kontrol edilmesi, örneğin,enerji yönetim sistemleri 
veya endüstriyel süreçlerin optimizasyonu.

Reinforcement Learning'in Zorlukları ve Geleceği

Keşif ve Sömürü Dengesi: Ajanın yeni şeyler denemesi (keşif) ile öğrendiği en iyi 
eylemleri gerçekleştirmesi (sömürü) arasında bir denge kurması gerekir.
Ölçeklenebilirlik: Büyük ve karmaşık durum uzaylarında verimli öğrenme sağlamak 
zor olabilir.
Gerçek Zamanlı Öğrenme: Özellikle robotik ve otonom sistemlerde, gerçek zamanlı öğrenme 
ve adaptasyon gereklidir.
Reinforcement Learning, makine öğrenmesi ve yapay zekanın en dinamik ve hızla gelişen 
alanlarından biridir. Geliştirilen yeni algoritmalar ve daha güçlü hesaplama kaynakları 
sayesinde RL, giderek daha karmaşık ve çeşitli problemleri çözmekte kullanılmaktadır.

"""


"""

A/B Testi
A/B testi, iki farklı versiyonun performansını karşılaştırmak amacıyla kullanılan bir 
deneysel yaklaşımdır. Genellikle kullanıcı deneyimi, pazarlama kampanyaları veya ürün 
özelliklerinin etkinliğini değerlendirmek için kullanılır.

A/B Testinin Temel Adımları:

Hipotez Belirleme: Deneyin neyi ölçmeyi amaçladığını ve başarı kriterlerini belirleyin.

Grupların Oluşturulması: Kullanıcıları rastgele iki gruba ayırın: 
A grubu (kontrol grubu) ve B grubu (deney grubu).
Değişkenin Uygulanması: A grubu mevcut durumu (kontrol) alırken, 
B grubuna yeni değişiklik (deney) uygulanır.

Veri Toplama: Her iki gruptan performans verileri toplanır.
Analiz ve Karşılaştırma: Grupların performans metrikleri karşılaştırılarak 
hangi versiyonun daha iyi olduğu belirlenir.

A/B Testinin Kullanım Alanları:
Web sitesi tasarımında farklı düzenlerin karşılaştırılması.
Pazarlama kampanyalarının etkisini ölçme.
Fiyatlandırma stratejilerinin karşılaştırılması.
Yeni ürün özelliklerinin kullanıcılar üzerindeki etkisini değerlendirme.

One-Armed Bandit (Tek Kollu Haydut)
One-Armed Bandit (Tek Kollu Haydut), kumarhanelerdeki slot makineleri için kullanılan 
bir terimdir. Ancak, makine öğrenmesi ve karar verme teorisinde, 
aynı terim bir tür çok kollu haydut (multi-armed bandit) probleminden türetilmiştir. 
Bu problemler, keşif (exploration) ve sömürü (exploitation) arasında bir denge kurmayı 
gerektirir.

Çok Kollu Haydut Problemi:
Bağlam: Bir kumarhanedeki slot makinelerinin her biri farklı bir kazanma oranına sahiptir 
ve amaç, hangi makinenin en yüksek ödül oranına sahip olduğunu bulmaktır.

Karar Problemi: Sınırlı denemelerle hangi makineyi oynayacağınıza karar vermek ve toplam 
ödülü maksimize etmeye çalışmak.

Çok Kollu Haydut (Multi-Armed Bandit) Algoritmaları:
Epsilon-Greedy: Çoğunlukla en iyi bilinen makineyi oynar (%1-ε), ancak %ε olasılıkla 
rastgele bir makine seçer.

UCB (Upper Confidence Bound): Her makine için bir güven aralığı hesaplar ve bu aralığın 
üst sınırını maksimize eden makineyi seçer.

Thompson Sampling: Her makinenin ödül dağılımını modellemek için Bayesian güncellemelerini 
kullanır ve her adımda bu dağılımlardan örnekler alarak makineyi seçer.
Çok Kollu Haydut Probleminin Kullanım Alanları:

Online reklam yerleştirme.
Hisse senedi portföy optimizasyonu.
Klinik deneylerde tedavi seçeneklerinin karşılaştırılması.
Kullanıcı kişiselleştirme ve öneri sistemleri.

A/B Testi ve Çok Kollu Haydut Arasındaki Farklar:

A/B Testi: İki sabit versiyonun performansını doğrudan karşılaştırır ve hangisinin daha 
iyi olduğunu belirlemek için genellikle sabit bir süre boyunca veri toplar.

Çok Kollu Haydut: Sürekli bir öğrenme süreci içerir ve denemelerden sürekli olarak veri 
toplayarak en iyi seçeneği bulmaya çalışır. Daha dinamik ve adaptif bir yaklaşımdır.
Bu iki yöntem de kullanıcı davranışlarını anlamak, performansı optimize etmek ve veri 
odaklı kararlar almak için güçlü araçlardır, ancak kullanılacak yöntem, 
spesifik kullanım durumuna ve hedeflere bağlı olarak seçilmelidir.


"""










