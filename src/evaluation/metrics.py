import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from collections import defaultdict

def modularity(G, communities):
    """
    Tính toán độ đo modularity của phân hoạch cộng đồng
    
    Args:
        G: networkx.Graph
        communities: list of sets, mỗi set chứa các đỉnh trong một cộng đồng
    
    Returns:
        float: Giá trị modularity
    """
    if isinstance(communities, dict):
        # Chuyển từ dictionary sang danh sách các tập hợp
        comm_dict = defaultdict(set)
        for node, comm_id in communities.items():
            if isinstance(comm_id, (set, frozenset)):
                for c in comm_id:
                    comm_dict[c].add(node)
            else:
                comm_dict[comm_id].add(node)
        communities = list(comm_dict.values())
    
    return nx.algorithms.community.modularity(G, communities)

def conductance(G, communities):
    """
    Tính toán độ đo conductance cho mỗi cộng đồng và trung bình
    
    Args:
        G: networkx.Graph
        communities: list of sets, mỗi set chứa các đỉnh trong một cộng đồng
    
    Returns:
        dict: Dictionary chứa conductance cho mỗi cộng đồng và giá trị trung bình
    """
    if isinstance(communities, dict):
        # Chuyển từ dictionary sang danh sách các tập hợp
        comm_dict = defaultdict(set)
        for node, comm_id in communities.items():
            if isinstance(comm_id, (set, frozenset)):
                for c in comm_id:
                    comm_dict[c].add(node)
            else:
                comm_dict[comm_id].add(node)
        communities = list(comm_dict.values())
    
    results = {}
    
    # Tính conductance cho mỗi cộng đồng
    for i, comm in enumerate(communities):
        cut_size = nx.cut_size(G, comm)
        volume = sum(dict(G.degree(comm)).values())
        if volume == 0:
            results[f'community_{i}'] = 0
        else:
            results[f'community_{i}'] = cut_size / volume
    
    # Tính trung bình
    results['average'] = np.mean(list(results.values()))
    
    return results

def nmi(true_labels, pred_labels):
    """
    Tính toán Normalized Mutual Information giữa hai phân hoạch
    
    Args:
        true_labels: dict mapping node -> community id (ground truth)
        pred_labels: dict mapping node -> community id (predicted)
    
    Returns:
        float: Giá trị NMI
    """
    # Chuyển đổi từ dict sang list để phù hợp với sklearn
    nodes = sorted(list(set(true_labels.keys()) & set(pred_labels.keys())))
    
    true_comm = [true_labels[node] for node in nodes]
    pred_comm = [pred_labels[node] for node in nodes]
    
    return normalized_mutual_info_score(true_comm, pred_comm)

def ari(true_labels, pred_labels):
    """
    Tính toán Adjusted Rand Index giữa hai phân hoạch
    
    Args:
        true_labels: dict mapping node -> community id (ground truth)
        pred_labels: dict mapping node -> community id (predicted)
    
    Returns:
        float: Giá trị ARI
    """
    # Chuyển đổi từ dict sang list để phù hợp với sklearn
    nodes = sorted(list(set(true_labels.keys()) & set(pred_labels.keys())))
    
    true_comm = [true_labels[node] for node in nodes]
    pred_comm = [pred_labels[node] for node in nodes]
    
    return adjusted_rand_score(true_comm, pred_comm)

def overlapping_nmi(true_communities, pred_communities):
    """
    Tính toán Normalized Mutual Information cho cộng đồng chồng chéo
    
    Args:
        true_communities: list of sets, mỗi set chứa các đỉnh trong một cộng đồng ground truth
        pred_communities: list of sets, mỗi set chứa các đỉnh trong một cộng đồng predicted
    
    Returns:
        float: Giá trị Overlapping NMI
    """
    try:
        from cdlib import evaluation
        
        return evaluation.overlapping_normalized_mutual_information(true_communities, pred_communities)
    except ImportError:
        print("Không thể tính toán Overlapping NMI. Hãy đảm bảo bạn đã cài đặt thư viện cdlib.")
        return None

def omega_index(true_communities, pred_communities):
    """
    Tính toán Omega Index cho cộng đồng chồng chéo
    
    Args:
        true_communities: list of sets, mỗi set chứa các đỉnh trong một cộng đồng ground truth
        pred_communities: list of sets, mỗi set chứa các đỉnh trong một cộng đồng predicted
    
    Returns:
        float: Giá trị Omega Index
    """
    try:
        from cdlib import evaluation
        
        return evaluation.omega(true_communities, pred_communities)
    except ImportError:
        print("Không thể tính toán Omega Index. Hãy đảm bảo bạn đã cài đặt thư viện cdlib.")
        return None 