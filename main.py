
# gerekli kütüphaneleri import ediyoruz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
# bazı kütüphanelerin sürümden kaynaklı verdiği hataları görmezden gelmesi için
# aşağıdakı metodu kullanıyoruz
import warnings
warnings.filterwarnings("ignore")

#verilen .csv dosyasını okuma işlemini yapıyoruz
data = pd.read_csv("Final-data.csv")
new_data = data.iloc[:, data.columns != 'Unnamed: 0']

#okuduğumuz verilerin normalizasyon işlemini yapıyoruz
scaler = MinMaxScaler()
normalize_data = new_data.copy()

def minmax_normalizer(x):
    for columnName, columnData in x.items():
        x[columnName] = scaler.fit_transform(np.array(columnData).reshape(-1, 1))


minmax_normalizer(normalize_data)

'''
aşağıdaki işlemler kmeans algoritması için bize wcss ve cluster(küme) sayısı ile ilgili belirtilen
aralıkta(2,11) bize grafik çizdiriyor. çıkan grafikten elbow methodu ile optimal küme sayısı hakkında bilgi 
edebiliriz. Grafiğe baakrak uygun küme sayısını 3 olarak belirledim
'''
k = list(range(2,11))
euclidean_distances = []
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(normalize_data)
    euclidean_distances.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(k, euclidean_distances, 'go--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('ELBOW METHOD')


# kmeans algoritmamızı tanımlıyoruz
kMeansWith3Cluster = KMeans(n_clusters = 3)

# modeli eğitiyoruz
kMeansWith3Cluster.fit(normalize_data)

# tahmin değerlerimiz üretiyoruz
y_pred = kMeansWith3Cluster.fit_predict(normalize_data)
print(y_pred)

# y_pred değerlerimizi yeni sütünda depoluyoruz
data['Cluster'] = y_pred+1 #to start the cluster number from 1

# küme sayısına göre, oluşacak kümelerin merkez noktalarını buluyoruz
# ve grafik üzerinde bu noktaları gösteriyoruz.
centroids = kMeansWith3Cluster.cluster_centers_
centroids = pd.DataFrame(centroids, columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9'])
centroids.index = np.arange(1, len(centroids)+1) # Start the index from 1


plt.figure(figsize=(12,6))
sns.set_palette("pastel")
sns.scatterplot(x=data['a1'], y = data['a2'], hue=data['Cluster'], palette='bright')
plt.show()


'''
dosya açma (sonuc.txt) işlemini aşağıda gerçekleştiriyoruz
kayıtları, kayıtlara göre wcss, bcss ve dunn index, sonuc.txt dosyasına
yazdırıyoruz. sonrasında dosyayı kapatıyoruz.
'''

f = open('sonuc.txt', 'w+')

for inde in data.index:
    f.write('Kayit {}       kume: {}\n'.format(inde, data['Cluster'][inde]))

f.close()

f = open('sonuc.txt', 'a')

counts = data.groupby(['Cluster']).count().to_string()
f.write('\n' + counts)
f.write('\n WCSS: {}\n'.format(kMeansWith3Cluster.inertia_))
f.write('\n BCSS: {}\n'.format(kMeansWith3Cluster.n_iter_))
f.write('\n Dunn Index: {}\n'.format(kMeansWith3Cluster.n_iter_ / kMeansWith3Cluster.inertia_))
f.close()

