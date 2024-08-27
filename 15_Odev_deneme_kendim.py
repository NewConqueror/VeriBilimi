#kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing 

#veriyi yükleme
veriler = pd.read_csv("odev_tenis.csv")

#yapılacaklar
"""
napıcam bu verileri alıcam outlook u kategori den nümerik e çeviricem le + ohe
windy direkt label encoder yeter zaten 2 değeri var play ise bizim çıkışımız olucak o da le

sonra  geri eleme backward elimination ile bu verileri düzenlicez
"""

#outlook u al 0 1 2 diye değerlerini ver le ile  sunny rainy vs sonra oht ile sütunlara çevir
outlook = veriler.iloc[:,0:1].values

le = preprocessing.LabelEncoder()
outlook = le.fit_transform(outlook)

oht = preprocessing.OneHotEncoder()
yeni_outlook = oht.fit_transform(outlook)

#windy yi kategoriden false true vs nümerik e çevirme 1 0 gibi
windy = veriler.iloc[:,3:4].values
yeni_windy = le.fit_transform(windy)

# play i no yes ten 1 0 a çevirme
play = veriler[:,-1].values
yeni_play = le.fit_transform(play)


# df leri birleştirme işlemi outlook temperature humidity var
outlook_birlestirilmis = pd.concat(yeni_outlook,veriler.iloc[1:3],axis=1)

yeni_veriler = pd.concat(outlook_birlestirilmis,yeni_windy,axis=1)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(yeni_veriler,yeni_play,test_size=0.33,random_state = 5)

from sklearn.preprocessing import StandardScaler


# verileri standartize ettik scale ettik ekstra bu

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)


from sklearn.linear_model import LinearRegression

reg = LinearRegression()
# modeli eğittik
reg.fit(x_train,y_train)

# x test i tahmin etmesi için verdik tahmin değerimizde y çıktımız olacak bunu y_test ile kıyaslicaz ne kadar doğru diye

tahmin = reg.predict(x_test)

import statsmodels.api as sm

X = np.append( arr = np.ones(14,1).astype(int), values=yeni_veriler,axis=1)

X_l = yeni_veriler.iloc[:,[0,1,2,3,4,5]].values

X_l = np.array(X_l,dtype=float)

model = sm.OLS(yeni_play,X_l).fit()

print(model.summary())
