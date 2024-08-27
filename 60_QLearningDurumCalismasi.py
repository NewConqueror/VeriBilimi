import gym
import numpy as np
import random
from tqdm import tqdm

"""

Bu kod, Reinforcement Learning (Pekiştirmeli Öğrenme) yöntemlerinden Q-learning 
algoritmasını kullanarak OpenAI Gym'den "Taxi-v3" ortamında bir taksi ajanını eğitir ve 
ardından bu eğitilmiş ajanı test eder. 

tqdm: Döngülerin ilerlemesini görselleştirmek için kullanılır.

env: Taxi-v3 ortamı başlatılır. Bu ortamda bir taksi, bir yolcuyu bir yerden alıp hedefe 
bırakmakla görevlidir.
env.render(): Ortamın o anki durumu konsola yazdırılır.
"""

env = gym.make("Taxi-v3", render_mode = "ansi")
env.reset()
print(env.render())

"""
0: guney
1: kuzey
2: dogu
3: bati
4: yolcuyu almak
5: yolcuyu birak
"""

"""
action_space ve state_space: Ortamdaki olası eylemlerin ve durumların sayısını alır.
q_table: Q-learning algoritması için kullanılan Q-tablosu, başlangıçta sıfırlarla 
doldurulur. Bu tablo, belirli bir durumdayken hangi eylemi yapmanın en iyi olduğunu öğrenmek 
için kullanılır.
"""

action_space = env.action_space.n
state_space = env.observation_space.n

q_table = np.zeros((state_space, action_space))

"""
alpha: Öğrenme oranı. Ajanın yeni bilgiye ne kadar hızlı uyum sağlayacağını belirler.

gamma: İndirim oranı. Gelecekteki ödüllerin bugünkü değeri üzerindeki etkisini belirler.

epsilon: Epsilon değeri, ajanı eğitirken keşfetme (exploration) ve sömürü (exploitation) 
arasında denge kurmak için kullanılır.

Keşif (exploration): Ajan rastgele bir eylem seçer ve ortamı keşfeder. Bu durum, 
ajan yeni şeyler öğrenmek için kullanılır.
Sömürü (exploitation): Ajan, Q-tablosunda öğrendiği en iyi eylemi seçer.
Q-Tablosunun Güncellenmesi: Q-tablosu, ajan her adımda ödülleri ve gelecekteki en iyi 
tahmini ödülleri göz önünde bulundurarak güncellenir.
"""

alpha = 0.1 # learning rate
gamma = 0.6 # discount rate
epsilon = 0.1 # epsilon

for i in tqdm(range(1, 100001)):
    
    state, _ = env.reset()
    
    done = False
    
    while not done:
        
        if random.uniform(0,1) < epsilon:  # explore %10
            action = env.action_space.sample()
        else: # exploit 
            action = np.argmax(q_table[state])
    
        next_state, reward, done, info, _  = env.step(action)
        
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action]) 
        
        state = next_state

"""
Eğitim tamamlandığında Q-tablosu doldurulmuş olur ve ajan, taksi görevini nasıl en iyi 
şekilde yerine getireceğini öğrenir.
"""
        
print("Training finished")

# test
total_epoch, total_penalties = 0, 0
episodes = 100

for i in tqdm(range(episodes)):
    
    state, _ = env.reset()
    
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        
        action = np.argmax(q_table[state])
    
        next_state, reward, done, info, _  = env.step(action)
                
        state = next_state
        
        if reward == -10:
            penalties += 1
            
        epochs += 1
    
    total_epoch += epochs
    total_penalties += penalties
    
print("Result after {} episodes".format(episodes))
print("Average timesteps per episode: ",total_epoch/episodes)
print("Average penalties per espisode: ",total_penalties/episodes)

"""
Test aşamasında, ajan eğitilmiş Q-tablosunu kullanarak 100 bölüm boyunca 
görevini yerine getirir.
total_epoch: Her bölümde ajan tarafından yapılan toplam adım sayısını kaydeder.
total_penalties: Ajanın her bölümde aldığı ceza sayısını kaydeder. 
Ceza, yolcunun yanlış bir yere bırakılması veya yanlış bir eylem yapılması durumunda alınır.

Test süreci tamamlandıktan sonra, ajan tarafından ortalama kaç adımda (epochs) görevin 
tamamlandığı ve bölüm başına ortalama kaç ceza (penalties) aldığı yazdırılır.


Bu kod, Taxi-v3 ortamında bir taksi ajanın Q-learning algoritması ile eğitilmesini ve 
ardından bu eğitimi test ederek ajanı değerlendirmeyi amaçlar. Ajan, bu eğitim sürecinde en 
iyi eylemleri öğrenerek yolcuyu mümkün olan en az hata ile en kısa sürede hedefe ulaştırmaya 
çalışır. Test aşamasında, ajanın başarımını ölçerek performansı değerlendirilir.
"""