import networkx as nx
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
import community as community_louvain
import random

class DisjointCommunityDetection:
    """
    Lớp cơ sở cho các thuật toán phát hiện cộng đồng không chồng chéo
    """
    def __init__(self, G=None):
        self.G = G
    
    def set_graph(self, G):
        """
        Cài đặt đồ thị
        
        Args:
            G: networkx.Graph
        """
        self.G = G
    
    def detect(self, **kwargs):
        """
        Phát hiện cộng đồng
        
        Returns:
            dict: Mapping từ node ID -> community ID
        """
        raise NotImplementedError("Phương thức này cần được triển khai bởi lớp con")


class GirvanNewman(DisjointCommunityDetection):
    """
    Thuật toán Girvan-Newman dựa trên độ trung gian cạnh
    """
    def detect(self, num_communities=None, max_iter=100, **kwargs):
        """
        Phát hiện cộng đồng sử dụng thuật toán Girvan-Newman
        
        Args:
            num_communities: Số lượng cộng đồng mong muốn. Nếu None, thuật toán sẽ chọn số lượng tối ưu
            max_iter: Số lần lặp tối đa
            
        Returns:
            dict: Mapping từ node ID -> community ID
        """
        if not self.G:
            raise ValueError("Đồ thị chưa được cài đặt")
        
        # Tạo bản sao để không làm thay đổi đồ thị gốc
        G_copy = self.G.copy()
        
        communities_generator = nx.algorithms.community.girvan_newman(G_copy)
        
        # Nếu số lượng cộng đồng được chỉ định
        if num_communities is not None:
            for i, communities in enumerate(communities_generator):
                if i == num_communities - 1 or i >= max_iter:
                    communities = list(communities)
                    break
        else:
            # Nếu không chỉ định, chọn số lượng tối ưu dựa trên modularity
            best_communities = None
            best_modularity = -1
            
            for i, communities in enumerate(communities_generator):
                communities = list(communities)
                current_modularity = nx.algorithms.community.modularity(self.G, communities)
                
                if current_modularity > best_modularity:
                    best_modularity = current_modularity
                    best_communities = communities
                
                if i >= max_iter:
                    break
            
            communities = best_communities
        
        # Chuyển đổi sang dạng node -> community_id
        community_mapping = {}
        for i, community in enumerate(communities):
            for node in community:
                community_mapping[node] = i
        
        return community_mapping


class Louvain(DisjointCommunityDetection):
    """
    Thuật toán Louvain dựa trên tối ưu hóa modularity
    """
    def detect(self, resolution=1.0, **kwargs):
        """
        Phát hiện cộng đồng sử dụng thuật toán Louvain
        
        Args:
            resolution: Tham số phân giải, giá trị lớn hơn tạo ra nhiều cộng đồng nhỏ hơn
            
        Returns:
            dict: Mapping từ node ID -> community ID
        """
        if not self.G:
            raise ValueError("Đồ thị chưa được cài đặt")
        
        # Sử dụng python-louvain
        partition = community_louvain.best_partition(self.G, resolution=resolution)
        
        return partition


class SpectralCommunity(DisjointCommunityDetection):
    """
    Thuật toán phân cụm phổ (Spectral Clustering)
    """
    def detect(self, n_clusters=2, **kwargs):
        """
        Phát hiện cộng đồng sử dụng phân cụm phổ
        
        Args:
            n_clusters: Số lượng cộng đồng
            
        Returns:
            dict: Mapping từ node ID -> community ID
        """
        if not self.G:
            raise ValueError("Đồ thị chưa được cài đặt")
        
        # Lấy ma trận kề
        adj_matrix = nx.to_numpy_array(self.G)
        
        # Sử dụng scikit-learn's SpectralClustering
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', **kwargs)
        sc.fit(adj_matrix)
        
        # Lấy các labels
        labels = sc.labels_
        
        # Chuyển đổi sang dict
        nodes = list(self.G.nodes())
        community_mapping = {nodes[i]: labels[i] for i in range(len(nodes))}
        
        return community_mapping


class KMeansCommunity(DisjointCommunityDetection):
    """
    Thuật toán K-means cho phát hiện cộng đồng
    """
    def detect(self, n_clusters=2, **kwargs):
        """
        Phát hiện cộng đồng sử dụng K-means
        
        Args:
            n_clusters: Số lượng cộng đồng
            
        Returns:
            dict: Mapping từ node ID -> community ID
        """
        if not self.G:
            raise ValueError("Đồ thị chưa được cài đặt")
        
        # Lấy ma trận kề
        adj_matrix = nx.to_numpy_array(self.G)
        
        # Sử dụng scikit-learn's KMeans
        kmeans = KMeans(n_clusters=n_clusters, **kwargs)
        kmeans.fit(adj_matrix)
        
        # Lấy các labels
        labels = kmeans.labels_
        
        # Chuyển đổi sang dict
        nodes = list(self.G.nodes())
        community_mapping = {nodes[i]: labels[i] for i in range(len(nodes))}
        
        return community_mapping


class HierarchicalCommunity(DisjointCommunityDetection):
    """
    Thuật toán phân cụm phân cấp (Hierarchical Clustering)
    """
    def detect(self, n_clusters=2, linkage='ward', **kwargs):
        """
        Phát hiện cộng đồng sử dụng phân cụm phân cấp
        
        Args:
            n_clusters: Số lượng cộng đồng
            linkage: Phương pháp liên kết ('ward', 'complete', 'average', 'single')
            
        Returns:
            dict: Mapping từ node ID -> community ID
        """
        if not self.G:
            raise ValueError("Đồ thị chưa được cài đặt")
        
        # Lấy ma trận kề
        adj_matrix = nx.to_numpy_array(self.G)
        
        # Chuyển ma trận kề thành ma trận khoảng cách
        # Sử dụng 1 - adj để cạnh có trọng số cao tương ứng với khoảng cách nhỏ
        dist_matrix = 1 - adj_matrix
        
        # Sử dụng scikit-learn's AgglomerativeClustering
        hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, **kwargs)
        hc.fit(dist_matrix)
        
        # Lấy các labels
        labels = hc.labels_
        
        # Chuyển đổi sang dict
        nodes = list(self.G.nodes())
        community_mapping = {nodes[i]: labels[i] for i in range(len(nodes))}
        
        return community_mapping


class LabelPropagation(DisjointCommunityDetection):
    """
    Thuật toán lan truyền nhãn (Label Propagation)
    """
    def detect(self, max_iter=100, **kwargs):
        """
        Phát hiện cộng đồng sử dụng lan truyền nhãn
        
        Args:
            max_iter: Số lần lặp tối đa
            
        Returns:
            dict: Mapping từ node ID -> community ID
        """
        if not self.G:
            raise ValueError("Đồ thị chưa được cài đặt")
        
        communities = nx.algorithms.community.label_propagation_communities(self.G)
        
        # Chuyển đổi sang dict
        community_mapping = {}
        for i, community in enumerate(communities):
            for node in community:
                community_mapping[node] = i
        
        return community_mapping


class InfomapCommunity(DisjointCommunityDetection):
    """
    Thuật toán Infomap (cần cài đặt thư viện infomap)
    """
    def detect(self, **kwargs):
        """
        Phát hiện cộng đồng sử dụng Infomap
        
        Returns:
            dict: Mapping từ node ID -> community ID
        """
        if not self.G:
            raise ValueError("Đồ thị chưa được cài đặt")
        
        try:
            from cdlib import algorithms
            
            communities = algorithms.infomap(self.G)
            
            # Lấy các cộng đồng
            comm_list = communities.communities
            
            # Chuyển đổi sang dict
            community_mapping = {}
            for i, community in enumerate(comm_list):
                for node in community:
                    community_mapping[node] = i
            
            return community_mapping
        except ImportError:
            print("Không thể sử dụng Infomap. Hãy đảm bảo bạn đã cài đặt thư viện cdlib và infomap.")
            return {} 