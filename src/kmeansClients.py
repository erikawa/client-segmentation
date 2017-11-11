# Agrupamento de Perfis de Clientes

# Bibliotecas utilizadas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import renders as rs
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Importando o dataset
data = pd.read_csv('../resources/dataset/customers.csv')
data.drop(['Region', 'Channel'], axis = 1, inplace = True) 

# Escalona os dados utilizando logaritmo natural
log_data = np.log(data)

# Identificando os pontos de desvios de cada categoria
outliersList = []
outlierSizes = []
for feature in log_data.keys():
    
    # Q1 (25º percentil dos dados) para o atributo dado
    Q1 = np.percentile(log_data[feature], 25)
    
    # Q3 (75º percentil dos dados) para o atributo dado
    Q3 = np.percentile(log_data[feature], 75)
    
    # 1,5 vezes a variação interquartil
    step = 1.5*(Q3-Q1)
    
    # Mostra os pontos de desvios
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    
    #Adicionando automaticamente os outliers a uma lista para posterior remoção
    outlier = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))].index.tolist()
    outliersList += outlier
    outlierSizes.append(len(outlier))

# Plota o gráfico mostrando quantos pontos de desvios de cada categoria
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(9, 3))
features = ('Fresh', 'Milk', 'Grocery',	'Frozen',	'Detergents_Paper',	'Delicatessen')
error = [0,0,0,0,0,0]
y_pos = np.arange(len(features))
ax.barh(y_pos, outlierSizes, xerr=error, align='center', color='blue', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.invert_yaxis()
ax.set_xlabel('Pontos de desvios')
ax.set_title('Pontos de Desvios de cada categoria')
plt.show()
        
# Seleciona os pontos que devem ser removidos
outliers  = list(set(outliersList))

# Remove os pontos de desvios
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

# Aplica o PCA para cada categoria
pca = PCA(n_components = 6)
pca.fit(good_data)
# Gera o gráfico das dimensões
pca_results = rs.pca_results(good_data, pca)

# Aplica o PCA para duas dimenões
pca = PCA(n_components = 2)
pca.fit(good_data)

# Redus a dimensão dos dados
reduced_data = pca.transform(good_data)

# Dataset com os dados reduzidos
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

# Calcula o número ideal de clusters com o Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(reduced_data);
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('Número de Clusters')
plt.ylabel('WSS')
plt.show();

# Calcula o número ideal de clusters com o método do coeficiente de silhueta
for K in range(2,11):
    
    clusterer = KMeans(n_clusters = K, max_iter = 200, n_init = 1, init = 'random', random_state = 101)
    clusterer.fit(reduced_data)
    preds = clusterer.predict(reduced_data)
    centers = clusterer.cluster_centers_
    score = silhouette_score(reduced_data, preds)
    print('Coeficiente para {} clusters = {}'.format(K, score))

# Aplica o algoritmo K-Means para 2 clusters
clusterer = KMeans(n_clusters = 2)
clusterer.fit(reduced_data)
preds = clusterer.predict(reduced_data)
# Recupera os centros de cada cluster
centers = clusterer.cluster_centers_

# Plota as duas dimensões
reduced_data_test = reduced_data.values
plt.scatter(reduced_data_test[preds == 0, 0], reduced_data_test[preds == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(reduced_data_test[preds == 1, 0], reduced_data_test[preds == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
plt.scatter(centers[:, 0], centers[:, 1], s = 50, c = 'black', label = 'Centroids')
plt.title('Agrupamento dos clientes')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()

# Recupera os valores originais dos dois clusters
log_centers = pca.inverse_transform(centers)
true_centers = np.exp(log_centers)

# Mostra os valores originais dos centros 
segments = ['Cluster {}'.format(i) for i in range(1,len(centers)+1)]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)
