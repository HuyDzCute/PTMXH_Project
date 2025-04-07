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
    
    # Kiểm tra và chuyển đổi loại dữ liệu
    communities = [set(str(node) for node in community) for community in communities]
    
    try:
        # Trường hợp cộng đồng không chồng chéo
        return nx.algorithms.community.modularity(G, communities)
    except Exception as e:
        print(f"Không thể tính modularity tiêu chuẩn: {str(e)}")
        print("Chuyển sang sử dụng phương pháp tính modularity cho cộng đồng chồng chéo...")
        
        # Phương pháp dự phòng cho cộng đồng chồng chéo
        try:
            from cdlib import evaluation
            from cdlib import NodeClustering
            
            # Chuyển sang định dạng cdlib NodeClustering
            comm_list = [list(comm) for comm in communities]
            # Tạo đối tượng NodeClustering
            node_clustering = NodeClustering(communities=comm_list, graph=G, method_name="CONGA")
            return evaluation.newman_girvan_modularity(node_clustering).score
        except ImportError:
            print("Không thể sử dụng cdlib.evaluation. Tính modularity thủ công...")
            return _custom_modularity(G, communities)
        except Exception as e:
            print(f"Lỗi khi sử dụng cdlib: {str(e)}")
            return _custom_modularity(G, communities)

def _custom_modularity(G, communities):
    """
    Tính modularity cho cộng đồng chồng chéo theo phương pháp thủ công
    """
    m = G.number_of_edges()
    if m == 0:
        return 0
    
    Q = 0
    for comm in communities:
        # Kiểm tra xem tất cả các nút trong comm có trong đồ thị G không
        valid_nodes = [node for node in comm if node in G]
        subgraph = G.subgraph(valid_nodes)
        
        # Số cạnh trong cộng đồng
        e_in = subgraph.number_of_edges()
        
        # Tổng bậc của các nút trong cộng đồng
        total_degree = sum(dict(G.degree(valid_nodes)).values())
        
        # Tăng modularity
        Q += (e_in / m) - ((total_degree / (2 * m)) ** 2)
    
    return Q

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
    
    # Kiểm tra và chuyển đổi loại dữ liệu
    communities = [set(str(node) for node in community) for community in communities]
    
    results = {}
    valid_communities = []
    
    # Tính conductance cho mỗi cộng đồng
    for i, comm in enumerate(communities):
        # Kiểm tra xem tất cả các nút trong comm có trong đồ thị G không
        valid_nodes = [node for node in comm if node in G]
        
        if not valid_nodes:
            results[f'community_{i}'] = 0
            continue
        
        try:
            cut_size = nx.cut_size(G, valid_nodes)
            volume = sum(dict(G.degree(valid_nodes)).values())
            
            if volume == 0:
                results[f'community_{i}'] = 0
            else:
                conductance_value = cut_size / volume
                results[f'community_{i}'] = conductance_value
                valid_communities.append(i)
        except Exception as e:
            print(f"Lỗi khi tính conductance cho cộng đồng {i}: {str(e)}")
            results[f'community_{i}'] = 0
    
    # Tính trung bình
    if valid_communities:
        valid_values = [results[f'community_{i}'] for i in valid_communities]
        results['average'] = np.mean(valid_values) if valid_values else 0
    else:
        # Nếu không có cộng đồng hợp lệ, trung bình là 0
        results['average'] = 0
    
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