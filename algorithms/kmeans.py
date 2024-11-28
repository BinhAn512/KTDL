from sklearn.cluster import KMeans

def kmeans_algorithm(data):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)
    data['Cluster'] = kmeans.labels_
    return data.head().to_html()

