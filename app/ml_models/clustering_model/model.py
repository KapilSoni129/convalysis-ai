from sklearn.cluster import DBSCAN

class TextClusteringModel():
    def __init__(self):
        self.model = DBSCAN(eps=0.4, min_samples=2, metric='cosine')
    
    def calculate_clusters(self, embeddings):
        clustering = self.model.fit(embeddings)
        return clustering.labels_