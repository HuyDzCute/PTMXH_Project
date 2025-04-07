import networkx as nx
import os
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import zipfile
import random

class GraphDataset:
    """
    Lớp xử lý dữ liệu đồ thị
    """
    def __init__(self, data_dir="src/data/"):
        """
        Khởi tạo
        
        Args:
            data_dir: Thư mục dữ liệu
        """
        self.data_dir = data_dir
        self.external_dir = os.path.join(data_dir, "external")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.external_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def load_karate_club(self):
        """
        Tải dữ liệu Zachary's Karate Club
        
        Returns:
            networkx.Graph
        """
        G = nx.karate_club_graph()
        
        # Chuyển đổi nhãn thành chuỗi để tránh lỗi khi sử dụng các thuật toán
        G = nx.relabel_nodes(G, {i: str(i) for i in G.nodes()})
        
        return G
    
    def load_dolphins(self):
        """
        Tải dữ liệu mạng xã hội của cá heo
        
        Returns:
            networkx.Graph
        """
        dolphins_file = os.path.join(self.external_dir, "dolphins.edges")
        
        # Tải dữ liệu nếu chưa tồn tại
        if not os.path.exists(dolphins_file):
            try:
                # Thử URLs khác nhau
                urls = [
                    "https://raw.githubusercontent.com/rabbits99/networks/master/dolphins.edges",
                    "https://raw.githubusercontent.com/cxnzensei/community-detection/main/dolphins.edges",
                    "https://raw.githubusercontent.com/gephi/gephi-toolkit-demos/master/src/main/resources/org/gephi/toolkit/demos/resources/dolphins.edges"
                ]
                
                for url in urls:
                    try:
                        print(f"Đang thử tải dolphins.edges từ {url}")
                        urllib.request.urlretrieve(url, dolphins_file)
                        if os.path.exists(dolphins_file) and os.path.getsize(dolphins_file) > 0:
                            print("Tải thành công!")
                            break
                    except Exception as e:
                        print(f"Lỗi khi tải từ {url}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Không thể tải dữ liệu dolphins: {str(e)}")
                # Tạo dữ liệu mẫu nếu không tải được
                print("Tạo dữ liệu dolphin mẫu...")
                self._create_sample_dolphins_data(dolphins_file)
        
        # Kiểm tra xem file có tồn tại không
        if not os.path.exists(dolphins_file) or os.path.getsize(dolphins_file) == 0:
            print("Tạo dữ liệu dolphin mẫu...")
            self._create_sample_dolphins_data(dolphins_file)
        
        # Tải đồ thị
        G = nx.read_edgelist(dolphins_file)
        
        return G
    
    def _create_sample_dolphins_data(self, output_file):
        """
        Tạo dữ liệu dolphins mẫu nếu không tải được dữ liệu thật
        """
        # Tạo đồ thị dolphin mẫu
        G = nx.random_partition_graph([10, 15, 7], 0.3, 0.05, seed=42)
        G = nx.relabel_nodes(G, {i: str(i) for i in G.nodes()})
        
        # Lưu vào file
        with open(output_file, 'w') as f:
            for u, v in G.edges():
                f.write(f"{u} {v}\n")
    
    def load_football(self):
        """
        Tải dữ liệu mạng football
        
        Returns:
            networkx.Graph và dict mapping node -> ground truth community
        """
        football_file = os.path.join(self.external_dir, "football.gml")
        
        # Tải dữ liệu nếu chưa tồn tại
        if not os.path.exists(football_file):
            url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"
            zip_file = os.path.join(self.external_dir, "football.zip")
            urllib.request.urlretrieve(url, zip_file)
            
            # Giải nén
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.external_dir)
            
            # Xóa file zip
            os.remove(zip_file)
        
        # Tải đồ thị
        G = nx.read_gml(football_file)
        
        # Lấy ground truth communities
        communities = {}
        for node, attr in G.nodes(data=True):
            communities[node] = attr['value']
        
        return G, communities
    
    def load_facebook_social_circles(self):
        """
        Tải dữ liệu Facebook Social Circles
        
        Returns:
            networkx.Graph
        """
        facebook_file = os.path.join(self.external_dir, "facebook_combined.txt")
        
        # Tải dữ liệu nếu chưa tồn tại
        if not os.path.exists(facebook_file):
            url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
            gz_file = os.path.join(self.external_dir, "facebook_combined.txt.gz")
            urllib.request.urlretrieve(url, gz_file)
            
            # Giải nén
            import gzip
            import shutil
            with gzip.open(gz_file, 'rb') as f_in:
                with open(facebook_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Xóa file gz
            os.remove(gz_file)
        
        # Tải đồ thị
        G = nx.read_edgelist(facebook_file, nodetype=int)
        
        # Chuyển đổi nhãn thành chuỗi để tránh lỗi khi sử dụng các thuật toán
        G = nx.relabel_nodes(G, {i: str(i) for i in G.nodes()})
        
        return G
    
    def load_email_eu(self):
        """
        Tải dữ liệu Email-Eu-core
        
        Returns:
            networkx.Graph
        """
        email_file = os.path.join(self.external_dir, "email-Eu-core.txt")
        ground_truth_file = os.path.join(self.external_dir, "email-Eu-core-department-labels.txt")
        
        # Tải dữ liệu nếu chưa tồn tại
        if not os.path.exists(email_file):
            url = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"
            gz_file = os.path.join(self.external_dir, "email-Eu-core.txt.gz")
            urllib.request.urlretrieve(url, gz_file)
            
            # Giải nén
            import gzip
            import shutil
            with gzip.open(gz_file, 'rb') as f_in:
                with open(email_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Xóa file gz
            os.remove(gz_file)
        
        # Tải dữ liệu ground truth nếu chưa tồn tại
        if not os.path.exists(ground_truth_file):
            url = "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz"
            gz_file = os.path.join(self.external_dir, "email-Eu-core-department-labels.txt.gz")
            urllib.request.urlretrieve(url, gz_file)
            
            # Giải nén
            import gzip
            import shutil
            with gzip.open(gz_file, 'rb') as f_in:
                with open(ground_truth_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Xóa file gz
            os.remove(gz_file)
        
        # Tải đồ thị
        G = nx.read_edgelist(email_file, nodetype=int)
        
        # Chuyển đổi nhãn thành chuỗi để tránh lỗi khi sử dụng các thuật toán
        G = nx.relabel_nodes(G, {i: str(i) for i in G.nodes()})
        
        # Tải ground truth communities
        communities = {}
        with open(ground_truth_file, 'r') as f:
            for line in f:
                node, community = line.strip().split()
                communities[str(node)] = int(community)
        
        return G, communities
    
    def generate_synthetic_data(self, n=100, k=4, p_in=0.3, p_out=0.05, seed=42):
        """
        Tạo dữ liệu tổng hợp sử dụng mô hình LFR
        
        Args:
            n: Số lượng đỉnh
            k: Số lượng cộng đồng
            p_in: Xác suất có cạnh trong cùng cộng đồng
            p_out: Xác suất có cạnh giữa các cộng đồng khác nhau
            seed: Giá trị khởi tạo cho bộ sinh số ngẫu nhiên
            
        Returns:
            networkx.Graph và dict mapping node -> community
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Tạo đồ thị ban đầu
        G = nx.Graph()
        
        # Thêm đỉnh
        G.add_nodes_from([str(i) for i in range(n)])
        
        # Gán cộng đồng cho mỗi đỉnh
        communities = {}
        community_sizes = [n // k] * k
        
        # Phân phối các đỉnh còn lại
        remainder = n % k
        for i in range(remainder):
            community_sizes[i] += 1
        
        # Gán cộng đồng
        start_idx = 0
        for comm_id, size in enumerate(community_sizes):
            for i in range(start_idx, start_idx + size):
                communities[str(i)] = comm_id
            start_idx += size
        
        # Thêm cạnh
        for i in range(n):
            for j in range(i + 1, n):
                node_i = str(i)
                node_j = str(j)
                
                # Xác định xác suất có cạnh
                if communities[node_i] == communities[node_j]:
                    p = p_in
                else:
                    p = p_out
                
                # Thêm cạnh với xác suất p
                if random.random() < p:
                    G.add_edge(node_i, node_j)
        
        return G, communities
    
    def generate_overlapping_communities(self, n=100, k=4, overlap_size=10, p_in=0.3, p_out=0.05, seed=42):
        """
        Tạo dữ liệu tổng hợp với cộng đồng chồng chéo
        
        Args:
            n: Số lượng đỉnh
            k: Số lượng cộng đồng
            overlap_size: Số lượng đỉnh chồng chéo giữa các cộng đồng
            p_in: Xác suất có cạnh trong cùng cộng đồng
            p_out: Xác suất có cạnh giữa các cộng đồng khác nhau
            seed: Giá trị khởi tạo cho bộ sinh số ngẫu nhiên
            
        Returns:
            networkx.Graph và list of sets (mỗi set là một cộng đồng)
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Tạo đồ thị ban đầu
        G = nx.Graph()
        
        # Thêm đỉnh
        G.add_nodes_from([str(i) for i in range(n)])
        
        # Tạo danh sách các cộng đồng
        communities = [set() for _ in range(k)]
        
        # Phân phối các đỉnh vào các cộng đồng
        remaining_nodes = set([str(i) for i in range(n)])
        
        # Phân phối các đỉnh chồng chéo
        overlap_nodes = set(random.sample(list(remaining_nodes), min(overlap_size * (k-1), n)))
        remaining_nodes -= overlap_nodes
        
        # Phân phối các đỉnh chồng chéo vào các cặp cộng đồng liên tiếp
        overlap_per_pair = overlap_size
        for i in range(k-1):
            if not overlap_nodes:
                break
                
            # Lấy các đỉnh cho cặp cộng đồng hiện tại
            pair_nodes = set(random.sample(list(overlap_nodes), 
                                         min(overlap_per_pair, len(overlap_nodes))))
            overlap_nodes -= pair_nodes
            
            # Thêm vào hai cộng đồng liên tiếp
            communities[i].update(pair_nodes)
            communities[i+1].update(pair_nodes)
        
        # Phân phối các đỉnh còn lại
        nodes_per_community = len(remaining_nodes) // k
        for i in range(k):
            if not remaining_nodes:
                break
                
            # Lấy các đỉnh cho cộng đồng hiện tại
            comm_nodes = set(random.sample(list(remaining_nodes), 
                                         min(nodes_per_community, len(remaining_nodes))))
            remaining_nodes -= comm_nodes
            
            # Thêm vào cộng đồng
            communities[i].update(comm_nodes)
        
        # Phân phối các đỉnh còn lại (nếu có)
        for i, node in enumerate(remaining_nodes):
            communities[i % k].add(node)
        
        # Thêm cạnh
        for i in range(n):
            for j in range(i + 1, n):
                node_i = str(i)
                node_j = str(j)
                
                # Xác định có cùng cộng đồng không
                same_community = False
                for comm in communities:
                    if node_i in comm and node_j in comm:
                        same_community = True
                        break
                
                # Xác định xác suất có cạnh
                if same_community:
                    p = p_in
                else:
                    p = p_out
                
                # Thêm cạnh với xác suất p
                if random.random() < p:
                    G.add_edge(node_i, node_j)
        
        return G, communities 