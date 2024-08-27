import gym
import random
import numpy as np

"""
Bu kod, Reinforcement Learning (Pekiştirmeli Öğrenme) kullanarak FrozenLake-v1 adlı bir 
ortamda bir ajan eğitir ve test eder. Bu ortam, gym kütüphanesinden alınan ve bir ajanı 
bir ızgara üzerinde buzlu bir gölette kaydırarak hedefe ulaşmasını sağlamayı amaçlayan 
basit bir simülasyondur. Bu ajan, eğitim süreci boyunca Q-learning algoritması kullanılarak 
öğrenir. 
"""

"""
environment: FrozenLake-v1 ortamı başlatılır. Burada is_slippery=False parametresi, 
ajan hareketlerinin daha deterministik olmasını sağlar.
nb_states ve nb_actions: Ortamdaki olası durumların ve eylemlerin sayısını alır.
qtable: Ajanın öğrenmesi için kullanılan Q-tablosu, başlangıçta sıfırlarla doldurulur. 
Bu tablo, ajan için belirli bir durumdayken hangi eylemi yapması gerektiğini 
öğrenmek için kullanılır.
"""

environment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q-table:")
print(qtable)

"""
action: Ortamdan rastgele bir eylem seçilir.
environment.step(action): Seçilen eylem ortamda uygulanır ve bu, yeni durumu (new_state), 
ödülü (reward), oyunun bitip bitmediğini (done) ve diğer bilgileri (info) döndürür.
"""

action = environment.action_space.sample()
"""
sol: 0
asagi: 1
sag: 2
yukari: 3
"""

# S1 -> (Action 1) -> S2
new_state, reward, done, info, _ = environment.step(action)

# %%
import gym
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt

environment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q-table:")
print(qtable) # ajanin beyni


"""
episodes: Ajanı eğitmek için çalıştırılacak bölüm sayısı.
alpha: Öğrenme oranı. Ajanın yeni bilgiye ne kadar hızlı uyum sağlayacağını belirler.
gamma: İndirim oranı. Gelecekteki ödüllerin bugünkü değeri üzerindeki etkisini belirler.
Eğitim sırasında, ajan rastgele veya öğrendiği Q-tablosuna dayalı bir eylem seçer ve 
bu eylemin sonucunda elde edilen ödül ile Q-tablosunu günceller.
outcomes: Her bölümün sonucunu (başarı veya başarısızlık) kaydeder.
"""


episodes = 1000 # episode
alpha = 0.5 # learning rate
gamma = 0.9 # discount rate

outcomes = []	

# training
for _ in tqdm(range(episodes)):
    
    state, _ = environment.reset()
    done = False # ajanin basari durumu 
    outcomes.append("Failure")
    
    while not done: # ajan basarili olana kadar state icerisinde hareket et (action sec ve uygula)
        
        # action
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()
            
        new_state, reward, done, info, _ = environment.step(action)
        
        # update q table
        qtable[state, action] =  qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        
        state = new_state
        
        if reward:
            outcomes[-1] = "Success"
        
print("Qtable After Training: ")  
print(qtable)

plt.bar(range(episodes), outcomes)

"""
Eğitim tamamlandıktan sonra Q-tablosu yazdırılır. Bu tablo, ajan için en iyi hareketleri 
gösterir.
plt.bar: Eğitim sürecindeki başarı ve başarısızlıkları görselleştirir.
"""
      
# test
episodes = 100 # episode
nb_success = 0

for _ in tqdm(range(episodes)):
    
    state, _ = environment.reset()
    done = False # ajanin basari durumu 
    
    while not done: # ajan basarili olana kadar state icerisinde hareket et (action sec ve uygula)
        
        # action
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()
            
        new_state, reward, done, info, _ = environment.step(action)
                
        state = new_state
        
        nb_success += reward
        
print("Success rate:", 100*nb_success/episodes)           
        
"""
Test aşaması: Eğitilmiş ajan, belirli sayıda bölümde test edilir ve başarı oranı hesaplanır.
nb_success: Ajanın başarılı olduğu bölümlerin sayısını tutar.
Başarı oranı: 100 bölüm üzerinden hesaplanır ve yüzde olarak verilir.


Bu kod, FrozenLake-v1 ortamında bir ajanı Q-learning algoritması kullanarak eğitir ve 
bu eğitimi bir başarı oranıyla test eder. Amaç, ajanı buzlu gölet üzerinde kaydırarak 
güvenli bir şekilde hedefe ulaştırmaktır. Eğitim sürecinde ajan, çevresindeki durumları 
gözlemleyerek ve ödülleri değerlendirerek en iyi hareket stratejisini öğrenir.
"""