#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


"""
iki sınıfı birbirinden bir fonksiyon  ile çizgi ile ayırmayı amaçlayan 
sınıflandırma algoritmasıdır Sigmoid vb
ikili sınıflandırma problemlerinde yaygın olarak kullanılır

Temel amacı, bağımlı değişkenin belirli bir sınıfa ait olma olasılığını tahmin etmektir. 
Bu nedenle, bağımlı değişken genellikle iki kategorilidir 
(örneğin, evet/hayır, 0/1, başarılı/başarısız).

Avantajları
Basit ve yorumlanabilir olması.
Eğitim ve tahmin süresinin hızlı olması.
İyi genelleştirme kabiliyeti ve overfitting (aşırı uyum) riskinin düşük olması.

Dezavantajları
Genellikle doğrusal olmayan ilişkileri iyi modelleyemez.
Sadece iki sınıflı problemler için uygundur 
(çok sınıflı problemler için genişletilmiş versiyonları olsa da).
Veri setinde yeterince örnek olmadığı durumlarda düşük performans gösterebilir.

Uygulama Alanları
Hastalık teşhisi (örneğin, bir hastanın belirli bir hastalığı olup olmadığını tahmin etme).
Kanser ya da değil
Finansal durum analizi (örneğin, bir müşterinin kredi alıp alamayacağını tahmin etme).
Alabilir Alamaz
Pazarlama (örneğin, bir müşterinin belirli bir ürünü satın alıp almayacağını tahmin etme).
Alır Almaz

bu çizginin sağında kalanlar erkek solunda kalanlar kadındır veya
bu çizginin üstünden kalanlar erkek altında kalanlar kadındır 
gibi çizgi ile sınıf ayrımı yapar

"""

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

"""
Confusion Matrix (Karışıklık Matrisi), sınıflandırma problemlerinde modelin performansını 
değerlendirmek için kullanılan bir araçtır. 
Bu matris, tahmin edilen sınıflarla gerçek sınıfların karşılaştırılmasını sağlar ve 
dört temel metriği içerir: True Positive (TP), False Positive (FP), 
True Negative (TN), ve False Negative (FN).

True Positive (TP): Modelin doğru bir şekilde pozitif olarak sınıflandırdığı örnekler.
False Positive (FP): Modelin pozitif olarak sınıflandırdığı, ancak gerçekte negatif olan örnekler (Type I error).
True Negative (TN): Modelin doğru bir şekilde negatif olarak sınıflandırdığı örnekler.
False Negative (FN): Modelin negatif olarak sınıflandırdığı, ancak gerçekte pozitif olan örnekler (Type II error).

Performans Metrikleri
Confusion matrix kullanılarak çeşitli performans metrikleri hesaplanabilir:

Accuracy (Doğruluk):
Precision (Kesinlik):
Recall (Duyarlılık veya Sensitivite):
F1-Score:
Specificity (Özgüllük):

!!!
diagonal dakiler doğru sonuçlardır diğerleri yanlış sonuçları gösterir
ör 7 3
   2 8
bu bize toplam 20 tahminin 15 inin doğru 5 inin yanlış olduğunu gösterir
3 yanlış FN 2 yanlış FP  7 doğru TP 8 doğru TN dir
"""

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


"""

KNN K nearest Neighbor sınıflandırma algoritmasıdır
k = 3 için en yakın 3 komşuya bakar eğer 3 komşunun 2 si erkek se erkek deri
3 komşunun en yakın 2 si kadın ise kadın deriz k değeri değiştirilebilir

en yakın komşuları hangi sınıfta ise o da o sınıfa dahil olur


bir veri noktasının sınıfını veya değerini tahmin etmek için en yakın komşularının 
sınıflarına veya değerlerine bakar. Bu algoritmanın temel prensibi, 
benzer veri noktalarının genellikle benzer sınıflarda olmasıdır.

Uzaklık Ölçümü: Veri noktaları arasındaki uzaklığı hesaplamak için genellikle 
Euclidean mesafesi kullanılır. Ancak, Manhattan mesafesi, Minkowski mesafesi gibi 
diğer mesafe ölçümleri de kullanılabilir. 
yani mesafeyi nasıl ölçtüğün de önemli

KNN Algoritmasının Adımları
Veri Kümesini Hazırlama:
Uzaklıkları Hesaplama:
En Yakın Komşuları Bulma: 
Tahmin Yapma:
Sınıflandırma: 
Regresyon: En yakın komşuların ortalama değerini hesaplayın.
Sonuçları Değerlendirme: Model performansını doğruluk, kesinlik, duyarlılık gibi metriklerle değerlendirin.


KNN'nin Avantajları ve Dezavantajları

Avantajları:
Basit ve Kolay Anlaşılır: Algoritma, anlaşılması ve uygulanması kolaydır.
Parametre Azlığı: Sadece K değeri ve uzaklık ölçümü seçilmelidir.
Esneklik: Hem sınıflandırma hem de regresyon problemlerinde kullanılabilir.

Dezavantajları:
Yüksek Hesaplama Maliyeti: Büyük veri setlerinde tüm mesafeleri hesaplamak zaman alıcıdır.
Bellek Kullanımı: Tüm veri setini bellekte tutması gerektiği için bellek yoğun bir algoritmadır.
Özellik Ölçeklendirme: Uzaklık bazlı bir yöntem olduğu için özelliklerin aynı ölçekte olması
önemlidir. Bu nedenle, özelliklerin normalize edilmesi veya standartlaştırılması gereklidir.

scale etmene gerek yani

K Değerinin Seçimi: K değerinin doğru seçilmesi önemlidir. Yanlış bir K değeri, 
model performansını olumsuz etkileyebilir.

KNN Algoritmasının Uygulama Alanları
Hastalık Teşhisi: Hastaların semptomlarına göre hastalıkların sınıflandırılması.
Pazarlama: Müşterilerin satın alma alışkanlıklarına göre pazarlama stratejilerinin belirlenmesi.
Görüntü Tanıma: Görüntülerin sınıflandırılması ve tanınması.
Öneri Sistemleri: Kullanıcıların önceki davranışlarına göre ürün önerilerinde bulunulması.
"""


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)



"""
SVM Support Vector Machine SVR a benzer SVC Support Vector Classifier dır 
yani sınıflandırma için kullanılanı

amacımız çizgi çekerek ve maksimum marjin değeri ile 2 sınıfı birbirinden ayırmaktır
ama bu çizgiler Fonksiyona göre belirlenir

lineer polinom RBF üssel gibi kerneller veririz 

yani çizgiyi çizicez ama kafamıza göre değil yukarıdaki 4 fonksiyona göre

SVR a benzer çizgimizi çekeriz marjin değerlerini ayarları ve 2 sınıfı birbirinden ayırırız
Marjin sınır çizgileri üstünde olanlara support vectors denir

SVR dan farklı olarak marjin içinde nokta olmaması gerekir olursa bu hata kabul edilir

SVM'nin temel amacı, farklı sınıfları en iyi şekilde ayıran bir 
hiperdüzlem (decision boundary) bulmaktır.

SVM'nin Çalışma Prensibi
SVM, veri noktalarını sınıflandırmak için aşağıdaki adımları izler:

Hiperdüzlem Seçimi: SVM, iki sınıfı ayıran en iyi hiperdüzlemi bulmaya çalışır. 
Bu hiperdüzlem, iki sınıf arasındaki marjini maksimize eder.
Destek Vektörleri (Support Vectors): Sınıflar arasındaki en yakın veri noktalarıdır 
ve hiperdüzlemin belirlenmesinde kritik rol oynarlar.
Marjin (Margin): Hiperdüzlem ile en yakın veri noktaları arasındaki mesafedir. 
SVM, bu marjini maksimize etmeye çalışır.

SVM, lineer ayrılabilir veri kümeleri için ideal bir algoritmadır. 
Lineer ayrılabilirlik, verilerin düz bir çizgi veya düzlem ile mükemmel bir şekilde 
ayrılabilmesi anlamına gelir. Ancak, çoğu gerçek dünya verisi lineer olarak 
ayrılabilir değildir. Bu durumlarda SVM, kernel trick adı verilen bir yöntem kullanarak 
veriyi daha yüksek boyutlu bir uzaya dönüştürür. Bu sayede, 
lineer olmayan veriler bile ayrılabilir hale gelir.

-Kernel hilesi çizgi ile ayrılamayacak verilerde veri kümesini orta noktasından yukarı
doğru çeker ve diğer veriler ile arasında bir mesafe oluşturulur sonra sınıflandırmak
için düzlem konur 
2 boyutlu veri merkezden yukarı çekiyon 3 boyutlu oluyor sonra sınırı koyuyorsun
boyut arttırmış gibi yapıyorsun-

Kernel Trick
Kernel trick, veriyi daha yüksek boyutlu bir uzaya dönüştürmeden, 
bu dönüşümün sonucunu hesaplamayı sağlar. Böylece hesaplama maliyeti düşer ve veriler 
daha etkili bir şekilde ayrılabilir. Yaygın kullanılan kernel fonksiyonları şunlardır:

Linear Kernel: Doğrusal ayrılabilir veri kümeleri için uygundur.
Polynomial Kernel: Polinom fonksiyonlarını kullanarak veri noktalarını ayırır.
Radial Basis Function (RBF) Kernel: Genellikle Gaussian Kernel olarak da bilinir. Verileri dairesel bir biçimde ayırır ve en yaygın kullanılan kernel fonksiyonlarından biridir.
Sigmoid Kernel: Sigmoid fonksiyonunu kullanarak verileri ayırır.

SVM'nin Avantajları ve Dezavantajları

Avantajları:

Etkin Performans: Genellikle yüksek boyutlu veri setlerinde iyi performans gösterir.
Genelleme Yeteneği: Marjini maksimize ettiği için overfitting (aşırı uyum) riskini azaltır.
Esneklik: Farklı kernel fonksiyonları kullanılarak lineer olmayan veriler de sınıflandırılabilir.

Dezavantajları:

Hesaplama Maliyeti: Büyük veri setlerinde ve yüksek boyutlu verilerde 
eğitim süresi uzun olabilir.
Parametre Seçimi: Kernel tipi ve parametrelerinin seçimi, 
modelin başarısı için kritik öneme sahiptir.
Özellik Ölçeklendirme: Özelliklerin aynı ölçekte olması önemlidir, 
bu nedenle verilerin normalize edilmesi gerekebilir. - scale yani -

SVM'nin Uygulama Alanları
Metin ve Hiper Metin Sınıflandırma: E-posta spam filtresi, belge sınıflandırma.
Biyometrik Tanıma: Yüz tanıma, el yazısı tanıma.
Genomik: Gen veri analizi ve sınıflandırma.
Tıp: Hastalık teşhisi, medikal görüntü analizi.
Finans: Kredi risk analizi, borsa tahminleri.

"""

from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)



"""
Temelinde koşullu olasılık yatar veriyi kategorilere böler

metin sınıflandırma, spam tespiti, duygu analizi gibi alanlarda yaygın olarak kullanılır.
Naive Bayes algoritması, Bayes teoremi üzerine kuruludur ve "naive" (saf) olarak 
adlandırılmasının nedeni, özelliklerin birbirinden bağımsız olduğunu varsaymasıdır. 
Bu varsayım, gerçek dünyada nadiren geçerli olsa da, algoritmanın basit ve etkili bir 
şekilde çalışmasını sağlar.

Naive Bayes Sınıflandırıcı
Naive Bayes sınıflandırıcı, Bayes teoremini kullanarak sınıflandırma yapar ve şu adımları izler:

Ön Olasılıkları Hesaplama (Prior Probability): Her sınıf için ön olasılıkları 
(prior probabilities) hesaplar. Bu, her sınıfın veri setindeki oranını gösterir.

Koşullu Olasılıkları Hesaplama (Likelihood): Her özellik için her sınıfa ait koşullu olasılıkları 
hesaplar. Bu, belirli bir özelliğin belirli bir sınıfta ortaya çıkma olasılığıdır.

Posterior Olasılıkları Hesaplama: Bayes teoremini kullanarak, bir veri noktasının 
her sınıfa ait olma olasılığını hesaplar.

Sınıflandırma:Posterior olasılıkları hesapladıktan sonra, en yüksek olasılığa sahip sınıf seçilir.


Naive Bayes Çeşitleri
Naive Bayes algoritmasının birkaç farklı türü vardır:

Gaussian Naive Bayes: Sürekli veri için kullanılır ve özelliklerin 
normal dağılım gösterdiğini varsayar.

Multinomial Naive Bayes: Özellikle metin sınıflandırma problemlerinde kullanılır 
ve özelliklerin (örneğin kelime frekanslarının) multinomial dağılım gösterdiğini varsayar.

Bernoulli Naive Bayes: İkili (binary) özellikler için kullanılır ve her özelliğin 
Bernoulli dağılımı gösterdiğini varsayar.

Avantajları ve Dezavantajları

Avantajları:
Basit ve Hızlı: Naive Bayes, öğrenmesi ve tahmin yapması hızlı bir algoritmadır.
Az Veriyle İyi Sonuçlar Verebilir: Küçük veri setlerinde bile etkili performans gösterebilir.
Çoklu Sınıf Problemleri İçin Uygundur: Birden fazla sınıfın olduğu problemlerde başarılıdır.

Dezavantajları:
Bağımsızlık Varsayımı: Özellikler arasındaki bağımsızlık varsayımı gerçekçi olmayabilir, 
bu da modelin performansını düşürebilir.
Sınıf Dengesi: Dengesiz sınıf dağılımlarında kötü performans gösterebilir.
Özelliklerin Skalası: Sürekli özellikler için, doğru sonuçlar elde etmek adına 
özelliklerin normalize edilmesi gerekebilir. -scale

Uygulama Alanları
E-posta Spam Filtreleme: Gelen e-postaların spam veya ham olup olmadığını belirleme.
Duygu Analizi: Metinlerin (örneğin, sosyal medya gönderilerinin) 
pozitif veya negatif duygu içerip içermediğini belirleme.
Doküman Sınıflandırma: Dokümanların belirli kategorilere 
(örneğin, spor, politika, teknoloji) ayrılması.
Hastalık Teşhisi: Hastaların belirtilerine göre belirli hastalıklara 
sahip olup olmadığını tahmin etme.

Özellikle yüksek boyutlu veri setlerinde ve metin madenciliği uygulamalarında 
yaygın olarak kullanılır.


!!!

Decision Tree ler ile birlikte güzel kullanılır
"""

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)



"""
Aynı Decision Tree sadece tahmin yerine bu sefer sınıflandırma yapıyor

Naive Bayesteki en yüksek Gain e yani kazanca göre ağaç oluşturur

Gain age 0.246 Gain income 0.029 Gain Student 0.151 Gain Credit Rating 0.048

burada en yüksek kazanç age te olduğu için en baştaki soru age sorusudur
ona göre agacı inşa eder sonra student sorusu gelir sonra credit en son income

4 tane sorudan fazla soramayız çünkü zaten 4 kıstasımız var


DecisionTreeClassifier, karar ağaçları (decision trees) kullanarak sınıflandırma yapan 
bir makine öğrenmesi algoritmasıdır. Karar ağaçları, 
veri kümesindeki örnekleri belirli özelliklerine göre dallara ayırarak sınıflandırma işlemi
gerçekleştirir. Bu yöntem, hem açıklanabilirliği hem de kolay anlaşılabilir olması nedeniyle
yaygın olarak kullanılır.

Karar Ağaçları (Decision Trees)
Karar ağaçları, kök düğümden başlayarak dallara ayrılan ve yaprak düğümlerde sonlanan 
bir yapıdır. Her düğümde, veri kümesi belirli bir özelliğe göre bölünür ve her dal, 
bu özelliğin belirli bir değer aralığına karşılık gelir. Son yaprak düğümler, 
sınıflandırılacak sınıfları temsil eder.


DecisionTreeClassifier Algoritmasının Çalışma Prensibi
Kök Düğüm (Root Node): Karar ağacının başlangıç noktasıdır. 
Tüm veri seti bu düğümde bulunur.
Bölünme (Splitting): Her düğümde, veri kümesi belirli bir özelliğe göre iki veya daha fazla
alt kümeye bölünür. Bölünme kriteri, veri kümesini en iyi şekilde ayıracak özelliği 
ve eşik değerini seçmeyi içerir. Bu, genellikle Gini impurity veya 
bilgi kazancı (information gain) gibi ölçütlerle yapılır.
Dallanma (Branching): Bölünme sonucunda oluşan alt kümeler, 
yeni düğümler oluşturur ve bu düğümlerden yeni dallar çıkar.
Yaprak Düğüm (Leaf Node): Veri kümesi artık daha fazla bölünemez hale geldiğinde veya 
belirli bir durdurma kriteri karşılandığında, yaprak düğüm oluşur. 
Bu düğümdeki veriler belirli bir sınıfa atanır.


Avantajları ve Dezavantajları

Avantajları:

Kolay Anlaşılabilirlik: Karar ağaçları, görselleştirilebileceği için kolay anlaşılır
ve yorumlanır.
Az Veri Ön İşleme Gereksinimi: Normalizasyon veya ölçekleme gibi ön işlemler gerektirmez.
Kategori ve Sürekli Veri ile Uyumlu: Hem kategorik hem de sürekli verilerle çalışabilir.

Dezavantajları:

Overfitting: Karar ağaçları, eğitim verisine fazla uyum sağlayarak overfitting yapabilir.
Bu, modelin genelleme yeteneğini azaltır.
Hesaplama Maliyeti: Büyük veri setlerinde ve çok sayıda özellik olduğunda, 
karar ağaçlarının oluşturulması zaman alabilir.
Dengesiz Veri Setlerine Duyarlılık: Dengesiz veri setlerinde, 
azınlık sınıflarını iyi öğrenemeyebilir.

Overfitting'i Önleme Yöntemleri
Budama (Pruning): Ağaç derinliğini sınırlayarak veya belirli kriterlere göre 
gereksiz dalları budayarak overfitting'i önler.
Ağaç Derinliğini Sınırlama: Maksimum derinlik, minimum yaprak sayısı gibi parametreler 
belirleyerek ağacın büyümesini kontrol eder.
Özellik Sayısını Sınırlama: Her bölünmede değerlendirilen maksimum özellik sayısını 
sınırlayarak overfitting'i azaltır.

Uygulama Alanları
Tıp: Hastalık teşhisi ve hasta sınıflandırması.
Finans: Kredi risk analizi ve müşteri segmentasyonu.
Pazarlama: Müşteri davranış analizi ve ürün önerileri.
Doğal Dil İşleme: Metin sınıflandırma ve duygu analizi.

"""

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)


"""
Bildiğimiz Random Forest 100 tane olsun 60 ı bir node a erkek 40 ı kadın diyorsa 
majority vote yani çoğunluk ne derse o yani erkek deriz

bireysel karar ağaçlarının tahminlerini birleştirerek daha doğru ve genelleştirilmiş 
tahminler elde etmeyi amaçlar. RandomForestClassifier, özellikle sınıflandırma 
problemlerinde yaygın olarak kullanılır ve karar ağaçlarının dezavantajlarını ortadan 
kaldırmak için tasarlanmıştır.



RandomForestClassifier Nasıl Çalışır?
RandomForestClassifier'ın çalışma prensibi birkaç ana adımdan oluşur:

Örnekleme (Bootstrap Sampling):

Eğitim veri setinden rastgele ve tekrarlı olarak örnekler seçilir. 
Bu işlem her ağaç için tekrarlanır, bu nedenle her bir karar ağacı 
farklı bir alt veri seti ile eğitilir.
Bu yöntem, "bootstrap aggregating" veya "bagging" olarak adlandırılır ve modelin genelleme 
yeteneğini artırır.


Özellik Seçimi (Feature Selection):

Her düğümde veri seti belirli bir özelliğe göre bölünürken, tüm özellikler arasından rastgele
bir alt küme seçilir ve en iyi bölünme bu alt küme üzerinden yapılır.
Bu rastgele özellik seçimi, ağaçlar arasındaki korelasyonu azaltır ve modelin çeşitliliğini
artırır.


Karar Ağaçlarının Oluşturulması (Tree Construction):

Her bir karar ağacı bağımsız olarak eğitim verisinin farklı alt kümeleri ve 
özellik alt kümeleri kullanılarak oluşturulur.
Ağaçlar tamamen büyütülür ve budama yapılmaz.


Tahminlerin Birleştirilmesi (Aggregation):

Sınıflandırma için, her bir ağacın sınıf tahminleri alınır ve en yaygın (majority voting) 
sınıf tahmini son tahmin olarak seçilir.
Regresyon için, her bir ağacın tahminlerinin ortalaması alınır.

RandomForestClassifier'ın Avantajları ve Dezavantajları

Avantajları:
Yüksek Doğruluk: Birçok karar ağacının tahminlerinin birleştirilmesi, modelin doğruluğunu 
ve genelleme yeteneğini artırır.

Overfitting'e Karşı Dayanıklılık: Bagging ve rastgele özellik seçimi, 
overfitting riskini azaltır.

Özelliklerin Önem Sıralaması: RandomForestClassifier, hangi özelliklerin tahminlerde 
daha önemli olduğunu belirleyebilir.

Esneklik: Hem sınıflandırma hem de regresyon problemlerinde kullanılabilir.
Paralel Hesaplama: Karar ağaçları bağımsız olarak oluşturulduğundan, 
paralel hesaplama ile hızlandırılabilir.

Dezavantajları:
Yüksek Hesaplama Maliyeti: Çok sayıda karar ağacı oluşturulması ve tahminlerin birleştirilmesi
hesaplama maliyetini artırabilir.
Yorumlanabilirlik:Tek bir karar ağacına kıyasla,random forest modeli daha az yorumlanabilirdir.


Hyperparametreler ve Ayarlamalar

n_estimators: atl ağaç sayısı.Daha fazla ağaç, genellikle daha iyi performans sağlar, 
ancak hesaplama maliyetini artırır.

Uygulama Alanları
Finans: Kredi risk analizi, dolandırıcılık tespiti.
Pazarlama: Müşteri segmentasyonu, müşteri sadakat analizi.
Tıp: Hastalık teşhisi, genetik veri analizi.
E-ticaret: Ürün önerileri, müşteri davranış analizi.
Doğal Dil İşleme: Metin sınıflandırma, duygu analizi.

"""

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)


"Accuracy paradoksu ZeroR algoritması "

"""
ROC eğrisi AUC eğrisi

ROC eğrisi, ikili sınıflandırma problemlerinde model performansını değerlendirmek için 
kullanılan grafiksel bir araçtır. ROC eğrisi, farklı eşik değerleri için modelin 
True Positive Rate (TPR) ve False Positive Rate (FPR) değerlerini gösterir.

ROC Eğrisi Nasıl Oluşturulur?

Tahmin Olasılıkları: Modelin pozitif sınıfa ait olasılık tahminleri kullanılarak başlar.
Eşik Değerlerinin Değiştirilmesi: Olasılık tahminleri için farklı eşik değerleri belirlenir.
TPR ve FPR Hesaplanması: Her eşik değeri için TPR ve FPR hesaplanır.
Grafik Oluşturulması: Yatay eksende FPR, dikey eksende TPR olacak şekilde bir grafik çizilir.
ROC eğrisi, FPR'ye karşı TPR'yi çizer ve (0,0) noktasından (1,1) noktasına kadar uzanır. 
Eğrinin altında kalan alan, modelin sınıflandırma performansını gösterir.


AUC (Area Under the Curve)
AUC, ROC eğrisinin altındaki alanı ifade eder ve modelin genel performansını 
tek bir sayısal değerle özetler. AUC değeri 0.5 ile 1 arasında değişir.

AUC = 1: Mükemmel sınıflandırıcı. Model, tüm pozitif ve negatif örnekleri doğru şekilde ayırır.
AUC = 0.5: Rastgele sınıflandırıcı. Model, pozitif ve negatif örnekleri rastgele tahmin eder.
AUC < 0.5: Kötü sınıflandırıcı. Model, çoğu tahmini yanlış yapar. 
Bu durum genellikle modelin kötü olduğunu gösterir ve tahminlerin tersine çevrilmesi durumunda anlamlı bir model elde edilebilir.

AUC'nin Yorumlanması
0.7-0.8: Kabul edilebilir performans.
0.8-0.9: Çok iyi performans.
0.9-1.0: Mükemmel performans.


ROC ve AUC'nin Avantajları ve Dezavantajları

Avantajları:
Eşik Değerlerinden Bağımsız: ROC eğrisi ve AUC, modelin performansını farklı 
eşik değerlerine karşı değerlendirdiği için eşik bağımsız bir ölçü sağlar.

Denge ve Genelleme: ROC ve AUC, dengesiz veri setlerinde bile modelin performansını 
etkili bir şekilde değerlendirebilir.

Dezavantajları:
Gerçek Dünya Senaryoları: ROC eğrisi, her iki sınıfın maliyetini veya faydasını 
hesaba katmaz, bu da bazı uygulamalarda gerçek dünya sonuçlarıyla tam uyumlu olmayabilir.
Hesaplama Maliyeti: Büyük veri setlerinde ROC eğrisinin ve AUC'nin hesaplanması 
zaman alıcı olabilir.

Uygulama Alanları
ROC eğrisi ve AUC, birçok farklı uygulama alanında kullanılır:

Tıp: Hastalık teşhisi, tanı testlerinin değerlendirilmesi.
Finans: Dolandırıcılık tespiti, kredi risk analizi.
Pazarlama: Müşteri sadakati, müşteri davranış analizi.
Doğal Dil İşleme: Metin sınıflandırma, duygu analizi.

"""

# 7. ROC , TPR, FPR değerleri 

y_proba = rfc.predict_proba(X_test)
print(y_test)
print(y_proba[:,0])

from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)








