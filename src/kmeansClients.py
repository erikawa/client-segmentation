# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import renders as rs
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Importing the dataset
data = pd.read_csv('../resources/dataset/customers.csv')
data.drop(['Region', 'Channel'], axis = 1, inplace = True) 

# TODO: Selecione três índices de sua escolha que você gostaria de obter como amostra do conjunto de dados
indices = [25, 170, 290]

# Crie um DataFrame das amostras escolhidas
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)

# TODO: Escalone os dados utilizando o algoritmo natural
log_data = np.log(data)

# TODO: Escalone a amostra de dados utilizando o algoritmo natural
log_samples = np.log(samples)

# Para cada atributo encontre os pontos de dados com máximos valores altos e baixos
outliersList = []
outlierSizes = []
for feature in log_data.keys():
    
    # TODO: Calcule Q1 (25º percentil dos dados) para o atributo dado
    Q1 = np.percentile(log_data[feature], 25)
    
    # TODO: Calcule Q3 (75º percentil dos dados) para o atributo dado
    Q3 = np.percentile(log_data[feature], 75)
    
    # TODO: Utilize a amplitude interquartil para calcular o passo do discrepante (1,5 vezes a variação interquartil)
    step = 1.5*(Q3-Q1)
    
    # Mostre os discrepantes
    #print "Data points considered outliers for the feature '{}':".format(feature)
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    
    #Adicionando automaticamente os outliers a uma lista para posterior remoção
    outlier = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))].index.tolist()
    outliersList += outlier
    outlierSizes.append(len(outlier))

plt.rcdefaults()
fig, ax = plt.subplots(figsize=(9, 3))
features = ('Fresh', 'Milk', 'Grocery',	'Frozen',	'Detergents_Paper',	'Delicatessen')
error = [0,0,0,0,0,0]
y_pos = np.arange(len(features))
ax.barh(y_pos, outlierSizes, xerr=error, align='center', color='blue', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Pontos de desvios')
ax.set_title('Pontos de Desvios de cada categoria')

plt.show()
        
# OPCIONAL: Selecione os índices dos pontos de dados que você deseja remover
outliers  = list(set(outliersList)) #Usando set para pegar os itens unicos da lista gerada no loop

# pegando os que se repetem em mais de um atributo
outliersCount = list(set([z for z in outliersList if outliersList.count(z) > 1]))
#print outliersCount

# Remova os discrepantes, caso nenhum tenha sido especificado
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

# TODO: Aplique a PCA ao ajustar os bons dados com o mesmo número de dimensões como atributos
pca = PCA(n_components = 6)
pca.fit(good_data)

# TODO: Transforme a amostra de data-log utilizando o ajuste da PCA acima
pca_samples = pca.transform(log_samples)
pca_results = rs.pca_results(good_data, pca)

# TODO: Aplique o PCA ao ajusta os bons dados com apenas duas dimensões
pca = PCA(n_components = 2)
pca.fit(good_data)

# TODO: Transforme os bons dados utilizando o ajuste do PCA acima
reduced_data = pca.transform(good_data)

# TODO: Transforme a amostre de log-data utilizando o ajuste de PCA acima
pca_samples = pca.transform(log_samples)

# Crie o DataFrame para os dados reduzidos
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

# achar o número ideal de clusters método Elbow
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

for K in range(2,7):
    # TODO: Aplique o algoritmo de clustering de sua escolha aos dados reduzidos 
    clusterer = KMeans(n_clusters = K, max_iter = 200, n_init = 1, init = 'random', random_state = 101)
    clusterer.fit(reduced_data)

    # TODO: Preveja o cluster para cada ponto de dado
    preds = clusterer.predict(reduced_data)

    # TODO: Ache os centros do cluster
    centers = clusterer.cluster_centers_


    # TODO: Preveja o cluster para cada amostra de pontos de dado transformados
    sample_preds = clusterer.predict(pca_samples)

    # TODO: Calcule a média do coeficiente de silhueta para o número de clusters escolhidos
    score = silhouette_score(reduced_data, preds)
    print('Coeficiente para {} clusters = {}'.format(K, score))

# Separando o código para o número ótimo de clusters

# TODO: Aplique o algoritmo de clustering de sua escolha aos dados reduzidos 
clusterer = KMeans(n_clusters = 5, max_iter = 200, n_init = 1, init = 'random', random_state = 101)
clusterer.fit(reduced_data)

# TODO: Preveja o cluster para cada ponto de dado
preds = clusterer.predict(reduced_data)

# TODO: Ache os centros do cluster
centers = clusterer.cluster_centers_


# TODO: Preveja o cluster para cada amostra de pontos de dado transformados
sample_preds = clusterer.predict(pca_samples)

# TODO: Calcule a média do coeficiente de silhueta para o número de clusters escolhidos
score = silhouette_score(reduced_data, preds)
reduced_data_test = reduced_data.values
# Visualising the clusters
plt.scatter(reduced_data_test[preds == 0, 0], reduced_data_test[preds == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(reduced_data_test[preds == 1, 0], reduced_data_test[preds == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
plt.scatter(reduced_data_test[preds == 2, 0], reduced_data_test[preds == 2, 1], s = 10, c = 'green', label = 'Cluster 3')
plt.scatter(reduced_data_test[preds == 3, 0], reduced_data_test[preds == 3, 1], s = 10, c = 'yellow', label = 'Cluster 4')
plt.scatter(reduced_data_test[preds == 4, 0], reduced_data_test[preds == 4, 1], s = 10, c = 'brown', label = 'Cluster 5')
plt.scatter(centers[:, 0], centers[:, 1], s = 50, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# TODO: Transforme inversamento os centros
log_centers = pca.inverse_transform(centers)

# TODO: Exponencie os centros
true_centers = np.exp(log_centers)

# Mostre os verdadeiros centros
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)
