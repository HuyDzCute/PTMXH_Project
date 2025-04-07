import networkx as nx
import numpy as np
import pandas as pd
import random
import os
from pathlib import Path

def load_graph_from_edgelist(file_path, delimiter=',', weighted=False, directed=False):
    """
    Tải đồ thị từ danh sách cạnh
    
    Args:
        file_path: Đường dẫn tới file danh sách cạnh
        delimiter: Ký tự phân cách trong file
        weighted: Đồ thị có trọng số hay không
        directed: Đồ thị có hướng hay không
        
    Returns:
        networkx.Graph hoặc networkx.DiGraph
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    try:
        if weighted:
            df = pd.read_csv(file_path, delimiter=delimiter, header=None)
            if df.shape[1] < 3:
                print("Cảnh báo: File không có cột trọng số, sử dụng trọng số 1.0 cho tất cả các cạnh")
                for _, row in df.iterrows():
                    G.add_edge(row[0], row[1], weight=1.0)
            else:
                for _, row in df.iterrows():
                    G.add_edge(row[0], row[1], weight=float(row[2]))
        else:
            df = pd.read_csv(file_path, delimiter=delimiter, header=None)
            for _, row in df.iterrows():
                G.add_edge(row[0], row[1])
    except Exception as e:
        print(f"Lỗi khi tải đồ thị: {e}")
        return None
        
    return G

def generate_random_graph(n, p, seed=None):
    """
    Tạo đồ thị ngẫu nhiên theo mô hình Erdős-Rényi
    
    Args:
        n: Số đỉnh
        p: Xác suất có cạnh giữa hai đỉnh
        seed: Giá trị khởi tạo cho bộ sinh số ngẫu nhiên
        
    Returns:
        networkx.Graph
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    return nx.erdos_renyi_graph(n, p, seed=seed)

def generate_LFR_benchmark(n, tau1, tau2, mu, average_degree=None, 
                           min_degree=None, max_degree=None, min_community=None, 
                           max_community=None, seed=None):
    """
    Tạo đồ thị LFR benchmark với cấu trúc cộng đồng
    
    Args:
        n: Số đỉnh
        tau1: Mũ cho phân phối bậc
        tau2: Mũ cho phân phối kích thước cộng đồng
        mu: Tham số trộn (tỷ lệ liên kết với các đỉnh ở cộng đồng khác)
        average_degree: Bậc trung bình
        min_degree: Bậc nhỏ nhất
        max_degree: Bậc lớn nhất
        min_community: Kích thước cộng đồng nhỏ nhất
        max_community: Kích thước cộng đồng lớn nhất
        seed: Giá trị khởi tạo cho bộ sinh số ngẫu nhiên
        
    Returns:
        networkx.Graph và dict ánh xạ đỉnh với cộng đồng
    """
    try:
        import networkx.generators.community as nxcommunity
        
        if average_degree is None:
            average_degree = int(0.1 * n)
        if min_degree is None:
            min_degree = max(int(average_degree / 2), 1)
        if max_degree is None:
            max_degree = min(int(average_degree * 2), n)
        if min_community is None:
            min_community = max(int(n / 20), 10)
        if max_community is None:
            max_community = max(int(n / 5), min_community + 10)
            
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        G = nxcommunity.LFR_benchmark_graph(
            n=n, tau1=tau1, tau2=tau2, mu=mu, average_degree=average_degree,
            min_degree=min_degree, max_degree=max_degree, 
            min_community=min_community, max_community=max_community,
            seed=seed
        )
        
        # Lấy thông tin cộng đồng
        communities = {node: frozenset(G.nodes[node]['community']) for node in G}
        
        return G, communities
    except ImportError:
        print("Không thể tạo đồ thị LFR benchmark. Hãy đảm bảo bạn đã cài đặt networkx phiên bản mới nhất.")
        return None, None

def get_adjacency_matrix(G):
    """
    Lấy ma trận kề của đồ thị
    
    Args:
        G: networkx.Graph
        
    Returns:
        numpy.ndarray
    """
    return nx.to_numpy_array(G)

def save_graph(G, file_path, format='edgelist'):
    """
    Lưu đồ thị
    
    Args:
        G: networkx.Graph
        file_path: Đường dẫn tới file
        format: Định dạng lưu (edgelist, adjlist, gexf, graphml)
    """
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if format == 'edgelist':
        nx.write_edgelist(G, file_path, data=['weight'])
    elif format == 'adjlist':
        nx.write_adjlist(G, file_path)
    elif format == 'gexf':
        nx.write_gexf(G, file_path)
    elif format == 'graphml':
        nx.write_graphml(G, file_path)
    else:
        print(f"Định dạng {format} không được hỗ trợ")

def get_real_world_datasets():
    """
    Trả về danh sách các bộ dữ liệu mạng xã hội thực tế phổ biến
    
    Returns:
        dict: Tên bộ dữ liệu và hàm tải tương ứng
    """
    return {
        "Zachary's Karate Club": nx.karate_club_graph,
        "Facebook Social Circles": lambda: nx.read_edgelist(
            "src/data/external/facebook_combined.txt", nodetype=int, create_using=nx.Graph()
        ),
        "Email Communication Network": lambda: nx.read_edgelist(
            "src/data/external/email-Eu-core.txt", nodetype=int, create_using=nx.Graph()
        )
    } 