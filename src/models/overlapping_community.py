import networkx as nx
import numpy as np
import random
from collections import defaultdict

class OverlappingCommunityDetection:
    """
    Lớp cơ sở cho các thuật toán phát hiện cộng đồng chồng chéo
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
            list of sets: Mỗi set chứa các đỉnh trong một cộng đồng
        """
        raise NotImplementedError("Phương thức này cần được triển khai bởi lớp con")


class CONGA(OverlappingCommunityDetection):
    """
    Thuật toán CONGA (Cluster-Overlap Newman Girvan Algorithm) 
    cho phát hiện cộng đồng chồng chéo
    """
    def detect(self, num_communities=None, max_split_degree=None, max_iter=100, **kwargs):
        """
        Phát hiện cộng đồng sử dụng thuật toán CONGA
        
        Args:
            num_communities: Số lượng cộng đồng mong muốn
            max_split_degree: Bậc tách tối đa. Nếu None, mặc định là |V| / 2
            max_iter: Số lần lặp tối đa
            
        Returns:
            list of sets: Mỗi set chứa các đỉnh trong một cộng đồng
        """
        if not self.G:
            raise ValueError("Đồ thị chưa được cài đặt")
        
        try:
            from cdlib import algorithms
            
            # Sử dụng cdlib để triển khai CONGA
            if max_split_degree is None:
                max_split_degree = len(self.G.nodes()) // 2
                
            communities = algorithms.conga(self.G, number_communities=num_communities, 
                                          max_communities=max_iter)
            
            return communities.communities
        except ImportError:
            print("Không thể sử dụng CONGA. Hãy đảm bảo bạn đã cài đặt thư viện cdlib.")
            # Nếu không có cdlib, trả về chia mặc định
            return self._fallback_overlapping_detection(num_communities)
    
    def _fallback_overlapping_detection(self, num_communities=2):
        """
        Phương pháp dự phòng nếu CONGA không khả dụng
        """
        # Sử dụng k-clique để phát hiện
        k = 3  # Kích thước clique
        communities_generator = nx.algorithms.community.k_clique_communities(self.G, k)
        communities = list(communities_generator)
        
        # Nếu không tìm thấy đủ cộng đồng, thêm một số đỉnh vào các cộng đồng hiện có
        if len(communities) < num_communities:
            # Tạo các cộng đồng ban đầu
            communities = [set(c) for c in communities]
            if not communities:
                # Nếu không tìm thấy cộng đồng nào, tạo một cộng đồng mới
                communities = [set(random.sample(list(self.G.nodes()), 
                                               len(self.G.nodes()) // num_communities))]
            
            # Thêm đỉnh vào các cộng đồng cho đến khi đạt số lượng mong muốn
            while len(communities) < num_communities:
                # Chọn một cộng đồng ngẫu nhiên
                source_comm = random.choice(communities)
                
                # Chọn một tập con ngẫu nhiên
                subset_size = len(source_comm) // 2
                if subset_size == 0:
                    subset_size = 1
                
                new_comm = set(random.sample(list(source_comm), subset_size))
                
                # Thêm một số đỉnh xung quanh
                for node in list(new_comm):
                    for neighbor in self.G.neighbors(node):
                        if random.random() < 0.3:  # 30% cơ hội thêm láng giềng
                            new_comm.add(neighbor)
                
                communities.append(new_comm)
        
        return communities


class CONGO(OverlappingCommunityDetection):
    """
    Thuật toán CONGO (CONGA Optimized)
    """
    def detect(self, num_communities=None, max_split_degree=None, max_iter=100, **kwargs):
        """
        Phát hiện cộng đồng sử dụng thuật toán CONGO
        
        Args:
            num_communities: Số lượng cộng đồng mong muốn
            max_split_degree: Bậc tách tối đa. Nếu None, mặc định là |V| / 2
            max_iter: Số lần lặp tối đa
            
        Returns:
            list of sets: Mỗi set chứa các đỉnh trong một cộng đồng
        """
        if not self.G:
            raise ValueError("Đồ thị chưa được cài đặt")
        
        try:
            from cdlib import algorithms
            
            # Sử dụng cdlib để triển khai CONGO
            if max_split_degree is None:
                max_split_degree = len(self.G.nodes()) // 2
                
            communities = algorithms.congo(self.G, number_communities=num_communities, 
                                         max_communities=max_iter)
            
            return communities.communities
        except ImportError:
            print("Không thể sử dụng CONGO. Hãy đảm bảo bạn đã cài đặt thư viện cdlib.")
            # Sử dụng cùng một phương pháp dự phòng như CONGA
            conga = CONGA(self.G)
            return conga._fallback_overlapping_detection(num_communities)


class COPRA(OverlappingCommunityDetection):
    """
    Thuật toán COPRA (Community Overlap PRopagation Algorithm)
    """
    def detect(self, v=2, max_iter=100, **kwargs):
        """
        Phát hiện cộng đồng sử dụng thuật toán COPRA
        
        Args:
            v: Số lượng cộng đồng tối đa mà một đỉnh có thể thuộc về
            max_iter: Số lần lặp tối đa
            
        Returns:
            list of sets: Mỗi set chứa các đỉnh trong một cộng đồng
        """
        if not self.G:
            raise ValueError("Đồ thị chưa được cài đặt")
        
        try:
            from cdlib import algorithms
            
            # Sử dụng cdlib để triển khai COPRA
            communities = algorithms.copra(self.G, v=v, max_iter=max_iter)
            
            return communities.communities
        except ImportError:
            print("Không thể sử dụng COPRA. Hãy đảm bảo bạn đã cài đặt thư viện cdlib.")
            # Sử dụng thuật toán khác làm dự phòng
            return self._fallback_label_propagation(v)
    
    def _fallback_label_propagation(self, v=2):
        """
        Triển khai đơn giản của thuật toán lan truyền nhãn cho cộng đồng chồng chéo
        """
        G = self.G
        nodes = list(G.nodes())
        n = len(nodes)
        
        # Khởi tạo: mỗi đỉnh có nhãn riêng
        node_labels = {}
        for i, node in enumerate(nodes):
            node_labels[node] = {i: 1.0}
        
        # Lặp cho đến khi hội tụ
        max_iter = 100
        for _ in range(max_iter):
            # Sao chép nhãn cũ
            old_labels = node_labels.copy()
            
            # Xử lý các đỉnh theo thứ tự ngẫu nhiên
            random.shuffle(nodes)
            
            for node in nodes:
                # Thu thập nhãn từ láng giềng
                neighbor_labels = defaultdict(float)
                for neighbor in G.neighbors(node):
                    for label, weight in old_labels[neighbor].items():
                        neighbor_labels[label] += weight
                
                # Nếu không có láng giềng, giữ nguyên nhãn
                if not neighbor_labels:
                    continue
                
                # Chọn v nhãn có trọng số cao nhất
                top_labels = dict(sorted(neighbor_labels.items(), 
                                        key=lambda x: x[1], reverse=True)[:v])
                
                # Chuẩn hóa trọng số
                total_weight = sum(top_labels.values())
                if total_weight > 0:
                    for label in top_labels:
                        top_labels[label] /= total_weight
                
                node_labels[node] = top_labels
            
            # Kiểm tra hội tụ
            converged = True
            for node in nodes:
                old_set = set(old_labels[node].keys())
                new_set = set(node_labels[node].keys())
                if old_set != new_set:
                    converged = False
                    break
            
            if converged:
                break
        
        # Nhóm các đỉnh theo nhãn
        communities = defaultdict(set)
        for node, labels in node_labels.items():
            for label in labels:
                communities[label].add(node)
        
        # Chuyển đổi thành list of sets
        return list(communities.values())


class BIGCLAM(OverlappingCommunityDetection):
    """
    Thuật toán BIGCLAM (Cluster Affiliation Model for Big Networks)
    """
    def detect(self, num_communities=2, **kwargs):
        """
        Phát hiện cộng đồng sử dụng thuật toán BIGCLAM
        
        Args:
            num_communities: Số lượng cộng đồng
            
        Returns:
            list of sets: Mỗi set chứa các đỉnh trong một cộng đồng
        """
        if not self.G:
            raise ValueError("Đồ thị chưa được cài đặt")
        
        try:
            from karateclub import BigClam
            
            # Chuyển đổi đồ thị NetworkX sang định dạng cạnh
            edges = list(self.G.edges())
            
            # Tạo ánh xạ giữa ID đỉnh gốc và ID đỉnh liên tục
            nodes = list(self.G.nodes())
            node_mapping = {node: i for i, node in enumerate(nodes)}
            
            # Chuyển đổi cạnh sang định dạng mới
            edge_list = [(node_mapping[u], node_mapping[v]) for u, v in edges]
            
            # Khởi tạo và chạy mô hình
            model = BigClam(dimensions=num_communities)
            model.fit(edge_list)
            
            # Lấy ma trận F (F[i, j] > 0 cho biết đỉnh i thuộc cộng đồng j)
            F = model.get_embedding()
            
            # Xây dựng cộng đồng
            communities = []
            for j in range(num_communities):
                # Lấy đỉnh có F[i, j] > 0
                comm = set()
                for i, node in enumerate(nodes):
                    if F[i, j] > 0:
                        comm.add(node)
                
                if comm:  # Chỉ thêm cộng đồng không rỗng
                    communities.append(comm)
            
            return communities
        except ImportError:
            print("Không thể sử dụng BIGCLAM. Hãy đảm bảo bạn đã cài đặt thư viện karateclub.")
            # Sử dụng thuật toán DEMON làm dự phòng
            return self._fallback_demon(num_communities)
    
    def _fallback_demon(self, num_communities=2):
        """
        Sử dụng thuật toán DEMON làm dự phòng
        """
        try:
            from cdlib import algorithms
            
            communities = algorithms.demon(self.G, epsilon=0.25)
            return communities.communities
        except ImportError:
            # Sử dụng k-clique làm dự phòng nếu cdlib không khả dụng
            k = 3  # Kích thước clique
            try:
                communities_generator = nx.algorithms.community.k_clique_communities(self.G, k)
                return list(communities_generator)
            except:
                print("Không thể phát hiện cộng đồng sử dụng k-clique.")
                return [] 