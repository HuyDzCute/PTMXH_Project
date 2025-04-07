import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import argparse
from pathlib import Path
import json
from tqdm import tqdm

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import GraphDataset
from src.models import (
    GirvanNewman, Louvain, SpectralCommunity, KMeansCommunity, 
    HierarchicalCommunity, LabelPropagation, InfomapCommunity
)
from src.models import (
    CONGA, CONGO, COPRA, BIGCLAM
)
from src.evaluation import modularity, conductance, nmi, ari, overlapping_nmi, omega_index
from src.visualization import visualize_communities, visualize_overlapping_communities

def get_disjoint_algorithms():
    """
    Trả về danh sách các thuật toán phát hiện cộng đồng không chồng chéo
    
    Returns:
        dict: Tên thuật toán và lớp tương ứng
    """
    return {
        "girvan_newman": GirvanNewman,
        "louvain": Louvain,
        "spectral": SpectralCommunity,
        "kmeans": KMeansCommunity,
        "hierarchical": HierarchicalCommunity,
        "label_propagation": LabelPropagation,
        "infomap": InfomapCommunity
    }

def get_overlapping_algorithms():
    """
    Trả về danh sách các thuật toán phát hiện cộng đồng chồng chéo
    
    Returns:
        dict: Tên thuật toán và lớp tương ứng
    """
    return {
        "conga": CONGA,
        "congo": CONGO,
        "copra": COPRA,
        "bigclam": BIGCLAM
    }

def get_available_datasets():
    """
    Trả về danh sách các bộ dữ liệu có sẵn
    
    Returns:
        list: Danh sách tên bộ dữ liệu
    """
    return [
        "karate_club",
        "dolphins",
        "football",
        "facebook",
        "email_eu",
        "synthetic",
        "synthetic_overlapping"
    ]

def evaluate_disjoint_algorithm(G, algorithm, true_communities=None, algorithm_params=None, visualize=False):
    """
    Đánh giá một thuật toán phát hiện cộng đồng không chồng chéo
    
    Args:
        G: networkx.Graph
        algorithm: Lớp thuật toán
        true_communities: dict hoặc list, ground truth communities (tùy chọn)
        algorithm_params: dict, tham số cho thuật toán (tùy chọn)
        visualize: bool, có hiển thị kết quả không
    
    Returns:
        dict: Kết quả đánh giá
    """
    if algorithm_params is None:
        algorithm_params = {}
    
    # Khởi tạo và chạy thuật toán
    start_time = time.time()
    
    algo = algorithm(G)
    communities = algo.detect(**algorithm_params)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Tính các chỉ số đánh giá
    mod = modularity(G, communities)
    cond = conductance(G, communities)
    
    # Tính NMI và ARI nếu có ground truth
    nmi_score = None
    ari_score = None
    if true_communities is not None:
        if isinstance(true_communities, list):
            # Chuyển từ list of sets sang dict cho NMI và ARI
            true_comm_dict = {}
            for i, comm in enumerate(true_communities):
                for node in comm:
                    true_comm_dict[node] = i
            true_communities = true_comm_dict
            
        # Lọc các đỉnh chung giữa true và predicted
        common_nodes = set(true_communities.keys()) & set(communities.keys())
        if common_nodes:
            filtered_true = {node: true_communities[node] for node in common_nodes}
            filtered_pred = {node: communities[node] for node in common_nodes}
            
            nmi_score = nmi(filtered_true, filtered_pred)
            ari_score = ari(filtered_true, filtered_pred)
    
    # Trực quan hóa nếu cần
    if visualize:
        visualize_communities(G, communities, title=f"{algorithm.__name__} - Modularity: {mod:.4f}")
    
    # Trả về kết quả
    result = {
        "algorithm": algorithm.__name__,
        "num_communities": len(set(communities.values())),
        "modularity": mod,
        "conductance_avg": cond["average"],
        "execution_time": execution_time,
        "nmi": nmi_score,
        "ari": ari_score
    }
    
    return result, communities

def evaluate_overlapping_algorithm(G, algorithm, true_communities=None, algorithm_params=None, visualize=False):
    """
    Đánh giá một thuật toán phát hiện cộng đồng chồng chéo
    
    Args:
        G: networkx.Graph
        algorithm: Lớp thuật toán
        true_communities: list of sets, ground truth communities (tùy chọn)
        algorithm_params: dict, tham số cho thuật toán (tùy chọn)
        visualize: bool, có hiển thị kết quả không
    
    Returns:
        dict: Kết quả đánh giá
    """
    if algorithm_params is None:
        algorithm_params = {}
    
    # Khởi tạo và chạy thuật toán
    start_time = time.time()
    
    algo = algorithm(G)
    communities = algo.detect(**algorithm_params)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Tính các chỉ số đánh giá
    mod = modularity(G, communities)
    cond = conductance(G, communities)
    
    # Tính Overlapping NMI và Omega Index nếu có ground truth
    onmi_score = None
    omega_score = None
    if true_communities is not None:
        # Nếu true_communities là dict, chuyển sang list of sets
        if isinstance(true_communities, dict):
            true_comm_dict = {}
            for node, comm_id in true_communities.items():
                if comm_id not in true_comm_dict:
                    true_comm_dict[comm_id] = set()
                true_comm_dict[comm_id].add(node)
            true_communities = list(true_comm_dict.values())
            
        onmi_score = overlapping_nmi(true_communities, communities)
        omega_score = omega_index(true_communities, communities)
    
    # Trực quan hóa nếu cần
    if visualize:
        visualize_overlapping_communities(G, communities, title=f"{algorithm.__name__} - Modularity: {mod:.4f}")
    
    # Trả về kết quả
    result = {
        "algorithm": algorithm.__name__,
        "num_communities": len(communities),
        "modularity": mod,
        "conductance_avg": cond["average"],
        "execution_time": execution_time,
        "overlapping_nmi": onmi_score,
        "omega_index": omega_score
    }
    
    return result, communities

def run_evaluation(dataset_name, algorithms="all", is_overlapping=False, visualize=False, output_dir="results"):
    """
    Chạy đánh giá các thuật toán trên một bộ dữ liệu
    
    Args:
        dataset_name: Tên bộ dữ liệu
        algorithms: List tên thuật toán hoặc "all" để chạy tất cả
        is_overlapping: Có phải đánh giá thuật toán cho cộng đồng chồng chéo không
        visualize: Có hiển thị kết quả không
        output_dir: Thư mục lưu kết quả
    
    Returns:
        pandas.DataFrame: Bảng kết quả đánh giá
    """
    # Tạo thư mục lưu kết quả
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tải dữ liệu
    dataset = GraphDataset()
    G = None
    true_communities = None
    
    if dataset_name == "karate_club":
        G = dataset.load_karate_club()
    elif dataset_name == "dolphins":
        G = dataset.load_dolphins()
    elif dataset_name == "football":
        G, true_communities = dataset.load_football()
    elif dataset_name == "facebook":
        G = dataset.load_facebook_social_circles()
    elif dataset_name == "email_eu":
        G, true_communities = dataset.load_email_eu()
    elif dataset_name == "synthetic":
        G, true_communities = dataset.generate_synthetic_data(n=100, k=5)
    elif dataset_name == "synthetic_overlapping":
        G, true_communities = dataset.generate_overlapping_communities(n=100, k=5, overlap_size=10)
    else:
        raise ValueError(f"Dataset {dataset_name} không được hỗ trợ")
    
    print(f"Đồ thị {dataset_name}: {G.number_of_nodes()} đỉnh, {G.number_of_edges()} cạnh")
    
    # Lựa chọn thuật toán
    if is_overlapping:
        all_algorithms = get_overlapping_algorithms()
    else:
        all_algorithms = get_disjoint_algorithms()
    
    if algorithms == "all":
        selected_algorithms = all_algorithms
    else:
        selected_algorithms = {name: all_algorithms[name] for name in algorithms if name in all_algorithms}
    
    # Chạy đánh giá
    results = []
    community_results = {}
    
    for name, algorithm in tqdm(selected_algorithms.items(), desc="Đánh giá thuật toán"):
        # Tùy chỉnh tham số cho từng thuật toán
        params = {}
        
        if name == "girvan_newman":
            params = {"num_communities": 5, "max_iter": 10}
        elif name == "spectral" or name == "kmeans" or name == "hierarchical":
            params = {"n_clusters": 5}
        elif name == "bigclam":
            params = {"num_communities": 5}
        
        try:
            if is_overlapping:
                result, communities = evaluate_overlapping_algorithm(
                    G, algorithm, true_communities, params, visualize
                )
            else:
                result, communities = evaluate_disjoint_algorithm(
                    G, algorithm, true_communities, params, visualize
                )
            
            results.append(result)
            community_results[name] = communities
            
            print(f"Thuật toán {name} đã hoàn thành với {result['num_communities']} cộng đồng")
            
        except Exception as e:
            print(f"Lỗi khi chạy thuật toán {name}: {e}")
    
    # Tạo DataFrame kết quả
    df_results = pd.DataFrame(results)
    
    # Lưu kết quả
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_type = "overlapping" if is_overlapping else "disjoint"
    
    # Lưu DataFrame
    df_results.to_csv(output_dir / f"{dataset_name}_{result_type}_{timestamp}.csv", index=False)
    
    # Lưu cấu trúc cộng đồng (để sử dụng sau này)
    for name, communities in community_results.items():
        if isinstance(communities, dict):
            # Chuyển dict thành danh sách để dễ lưu trữ
            comm_dict = {}
            for node, comm_id in communities.items():
                if isinstance(comm_id, (set, frozenset)):
                    comm_dict[str(node)] = list(comm_id)
                else:
                    comm_dict[str(node)] = comm_id
            
            with open(output_dir / f"{dataset_name}_{name}_{timestamp}.json", 'w') as f:
                json.dump(comm_dict, f)
        else:
            # Chuyển list of sets thành list of lists
            comm_list = [list(comm) for comm in communities]
            
            with open(output_dir / f"{dataset_name}_{name}_{timestamp}.json", 'w') as f:
                json.dump(comm_list, f)
    
    return df_results

def main():
    parser = argparse.ArgumentParser(description='Chạy đánh giá thuật toán phát hiện cộng đồng')
    
    # Tham số đầu vào
    parser.add_argument('--dataset', type=str, default="karate_club",
                        help=f'Bộ dữ liệu (mặc định: karate_club). Các lựa chọn: {", ".join(get_available_datasets())}')
    
    parser.add_argument('--algorithms', type=str, nargs='+', default="all",
                        help='Danh sách thuật toán hoặc "all" để chạy tất cả (mặc định: all)')
    
    parser.add_argument('--overlapping', action='store_true',
                        help='Đánh giá thuật toán phát hiện cộng đồng chồng chéo (mặc định: False)')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Hiển thị kết quả phát hiện cộng đồng (mặc định: False)')
    
    parser.add_argument('--output_dir', type=str, default="results",
                        help='Thư mục lưu kết quả (mặc định: results)')
    
    args = parser.parse_args()
    
    # Kiểm tra tham số
    if args.dataset not in get_available_datasets():
        print(f"Bộ dữ liệu {args.dataset} không được hỗ trợ")
        print(f"Các bộ dữ liệu có sẵn: {', '.join(get_available_datasets())}")
        sys.exit(1)
    
    # Chạy đánh giá
    results = run_evaluation(
        dataset_name=args.dataset,
        algorithms=args.algorithms,
        is_overlapping=args.overlapping,
        visualize=args.visualize,
        output_dir=args.output_dir
    )
    
    # Hiển thị kết quả
    print("\nKết quả đánh giá:")
    print(results)
    
    # Tạo biểu đồ so sánh
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    results.plot(x='algorithm', y='modularity', kind='bar', ax=plt.gca())
    plt.title('Modularity')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    results.plot(x='algorithm', y='execution_time', kind='bar', ax=plt.gca())
    plt.title('Thời gian thực thi (giây)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/comparison_{args.dataset}.png", dpi=300, bbox_inches='tight')
    
    print(f"\nKết quả đã được lưu vào thư mục {args.output_dir}")

if __name__ == "__main__":
    main() 