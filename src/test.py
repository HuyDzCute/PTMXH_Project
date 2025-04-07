import os
import sys
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from tqdm import tqdm
import argparse
from pathlib import Path

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

def run_comprehensive_test(output_dir="test_results"):
    """
    Chạy kiểm tra toàn diện trên nhiều bộ dữ liệu và thuật toán khác nhau
    
    Args:
        output_dir: Thư mục lưu kết quả
    """
    # Tạo thư mục lưu kết quả
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tải dữ liệu
    dataset = GraphDataset()
    
    # Chuẩn bị bộ dữ liệu
    datasets = {
        "karate_club": (dataset.load_karate_club(), None),
        "dolphins": (dataset.load_dolphins(), None),
        "football": dataset.load_football(),
        "synthetic_small": dataset.generate_synthetic_data(n=50, k=3, p_in=0.3, p_out=0.05),
        "synthetic_medium": dataset.generate_synthetic_data(n=200, k=5, p_in=0.25, p_out=0.01),
        "synthetic_overlapping": dataset.generate_overlapping_communities(n=100, k=4, overlap_size=10)
    }
    
    # Chuẩn bị thuật toán
    disjoint_algorithms = {
        "girvan_newman": GirvanNewman,
        "louvain": Louvain,
        "spectral": SpectralCommunity,
        "kmeans": KMeansCommunity,
        "hierarchical": HierarchicalCommunity,
        "label_propagation": LabelPropagation
    }
    
    overlapping_algorithms = {
        "conga": CONGA,
        "copra": COPRA,
        "bigclam": BIGCLAM
    }
    
    # Danh sách kết quả
    all_results = []
    
    # Chạy kiểm tra với thuật toán không chồng chéo
    for dataset_name, (G, true_communities) in tqdm(datasets.items(), desc="Đánh giá bộ dữ liệu"):
        print(f"\nBộ dữ liệu: {dataset_name} - {G.number_of_nodes()} đỉnh, {G.number_of_edges()} cạnh")
        
        # Chạy các thuật toán không chồng chéo
        for algo_name, algo_class in tqdm(disjoint_algorithms.items(), desc="Thuật toán không chồng chéo"):
            # Thiết lập tham số
            params = {}
            if algo_name == "girvan_newman":
                params = {"num_communities": 5, "max_iter": 10}
            elif algo_name in ["spectral", "kmeans", "hierarchical"]:
                params = {"n_clusters": 5}
            
            try:
                # Khởi tạo và chạy thuật toán
                start_time = time.time()
                
                algorithm = algo_class(G)
                communities = algorithm.detect(**params)
                
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
                        # Chuyển từ list of sets sang dict
                        true_comm_dict = {}
                        for i, comm in enumerate(true_communities):
                            for node in comm:
                                true_comm_dict[node] = i
                        true_communities = true_comm_dict
                    
                    # Lọc các đỉnh chung
                    common_nodes = set(true_communities.keys()) & set(communities.keys())
                    if common_nodes:
                        filtered_true = {node: true_communities[node] for node in common_nodes}
                        filtered_pred = {node: communities[node] for node in common_nodes}
                        
                        nmi_score = nmi(filtered_true, filtered_pred)
                        ari_score = ari(filtered_true, filtered_pred)
                
                # Thêm kết quả
                result = {
                    "dataset": dataset_name,
                    "algorithm": algo_name,
                    "overlapping": False,
                    "num_communities": len(set(communities.values())),
                    "modularity": mod,
                    "conductance_avg": cond["average"],
                    "execution_time": execution_time,
                    "nmi": nmi_score,
                    "ari": ari_score
                }
                
                all_results.append(result)
                
                print(f"  {algo_name}: {result['num_communities']} cộng đồng, mod={mod:.4f}, time={execution_time:.4f}s")
                
            except Exception as e:
                print(f"  Lỗi khi chạy {algo_name}: {e}")
        
        # Chạy các thuật toán cộng đồng chồng chéo
        for algo_name, algo_class in tqdm(overlapping_algorithms.items(), desc="Thuật toán chồng chéo"):
            # Thiết lập tham số
            params = {}
            if algo_name in ["conga", "congo"]:
                params = {"num_communities": 5}
            elif algo_name == "bigclam":
                params = {"num_communities": 5}
            
            try:
                # Khởi tạo và chạy thuật toán
                start_time = time.time()
                
                algorithm = algo_class(G)
                communities = algorithm.detect(**params)
                
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
                
                # Thêm kết quả
                result = {
                    "dataset": dataset_name,
                    "algorithm": algo_name,
                    "overlapping": True,
                    "num_communities": len(communities),
                    "modularity": mod,
                    "conductance_avg": cond["average"],
                    "execution_time": execution_time,
                    "overlapping_nmi": onmi_score,
                    "omega_index": omega_score
                }
                
                all_results.append(result)
                
                print(f"  {algo_name}: {result['num_communities']} cộng đồng, mod={mod:.4f}, time={execution_time:.4f}s")
                
            except Exception as e:
                print(f"  Lỗi khi chạy {algo_name}: {e}")
    
    # Tạo DataFrame kết quả
    df_results = pd.DataFrame(all_results)
    
    # Lưu kết quả
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    df_results.to_csv(output_dir / f"comprehensive_test_{timestamp}.csv", index=False)
    
    # Tạo báo cáo tóm tắt
    print("\nTóm tắt kết quả:")
    
    # Tạo bảng tóm tắt với modularity trung bình và thời gian chạy trung bình
    summary = df_results.groupby(['algorithm', 'overlapping']).agg({
        'modularity': 'mean',
        'conductance_avg': 'mean',
        'execution_time': 'mean',
        'nmi': 'mean',
        'ari': 'mean',
        'overlapping_nmi': 'mean',
        'omega_index': 'mean'
    }).reset_index()
    
    print(summary)
    
    # Lưu tóm tắt
    summary.to_csv(output_dir / f"summary_{timestamp}.csv", index=False)
    
    # Tạo biểu đồ so sánh
    plt.figure(figsize=(15, 10))
    
    # Modularity comparison
    plt.subplot(2, 2, 1)
    sns.barplot(x='algorithm', y='modularity', hue='overlapping', data=df_results)
    plt.title('Modularity trung bình trên các bộ dữ liệu')
    plt.xticks(rotation=45)
    
    # Execution time comparison
    plt.subplot(2, 2, 2)
    sns.barplot(x='algorithm', y='execution_time', hue='overlapping', data=df_results)
    plt.title('Thời gian thực thi trung bình (giây)')
    plt.xticks(rotation=45)
    
    # Number of communities comparison
    plt.subplot(2, 2, 3)
    sns.barplot(x='algorithm', y='num_communities', hue='overlapping', data=df_results)
    plt.title('Số lượng cộng đồng trung bình')
    plt.xticks(rotation=45)
    
    # Conductance comparison
    plt.subplot(2, 2, 4)
    sns.barplot(x='algorithm', y='conductance_avg', hue='overlapping', data=df_results)
    plt.title('Conductance trung bình')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    # Tạo biểu đồ so sánh riêng cho mỗi bộ dữ liệu
    for dataset_name in datasets.keys():
        dataset_results = df_results[df_results['dataset'] == dataset_name]
        
        if len(dataset_results) > 0:
            plt.figure(figsize=(15, 10))
            
            # Modularity comparison
            plt.subplot(2, 2, 1)
            sns.barplot(x='algorithm', y='modularity', hue='overlapping', data=dataset_results)
            plt.title(f'{dataset_name} - Modularity')
            plt.xticks(rotation=45)
            
            # Execution time comparison
            plt.subplot(2, 2, 2)
            sns.barplot(x='algorithm', y='execution_time', hue='overlapping', data=dataset_results)
            plt.title(f'{dataset_name} - Thời gian thực thi (giây)')
            plt.xticks(rotation=45)
            
            # Number of communities comparison
            plt.subplot(2, 2, 3)
            sns.barplot(x='algorithm', y='num_communities', hue='overlapping', data=dataset_results)
            plt.title(f'{dataset_name} - Số lượng cộng đồng')
            plt.xticks(rotation=45)
            
            # Conductance comparison
            plt.subplot(2, 2, 4)
            sns.barplot(x='algorithm', y='conductance_avg', hue='overlapping', data=dataset_results)
            plt.title(f'{dataset_name} - Conductance')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{dataset_name}_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    print(f"Kết quả đã được lưu vào thư mục {output_dir}")
    
    return df_results

def run_scaling_test(output_dir="test_results"):
    """
    Chạy kiểm tra khả năng mở rộng (scaling) của các thuật toán với đồ thị có kích thước khác nhau
    
    Args:
        output_dir: Thư mục lưu kết quả
    """
    # Tạo thư mục lưu kết quả
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tải dữ liệu
    dataset = GraphDataset()
    
    # Các kích thước đồ thị khác nhau
    sizes = [50, 100, 200, 500, 1000]
    
    # Chuẩn bị thuật toán
    algorithms = {
        "louvain": Louvain,
        "spectral": SpectralCommunity,
        "kmeans": KMeansCommunity,
        "label_propagation": LabelPropagation,
        "conga": CONGA
    }
    
    # Danh sách kết quả
    all_results = []
    
    # Chạy kiểm tra
    for n in tqdm(sizes, desc="Kích thước đồ thị"):
        # Tạo đồ thị tổng hợp
        G, _ = dataset.generate_synthetic_data(n=n, k=int(np.sqrt(n)), p_in=0.3, p_out=0.05)
        
        print(f"\nĐồ thị kích thước n={n}: {G.number_of_nodes()} đỉnh, {G.number_of_edges()} cạnh")
        
        # Chạy các thuật toán
        for algo_name, algo_class in tqdm(algorithms.items(), desc="Thuật toán"):
            # Thiết lập tham số
            params = {}
            if algo_name in ["spectral", "kmeans"]:
                params = {"n_clusters": int(np.sqrt(n))}
            elif algo_name == "conga":
                params = {"num_communities": int(np.sqrt(n))}
            
            try:
                # Khởi tạo và chạy thuật toán
                start_time = time.time()
                
                algorithm = algo_class(G)
                communities = algorithm.detect(**params)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Tính các chỉ số đánh giá
                mod = modularity(G, communities)
                
                # Thêm kết quả
                result = {
                    "size": n,
                    "algorithm": algo_name,
                    "execution_time": execution_time,
                    "modularity": mod
                }
                
                all_results.append(result)
                
                print(f"  {algo_name}: time={execution_time:.4f}s, mod={mod:.4f}")
                
            except Exception as e:
                print(f"  Lỗi khi chạy {algo_name}: {e}")
    
    # Tạo DataFrame kết quả
    df_results = pd.DataFrame(all_results)
    
    # Lưu kết quả
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    df_results.to_csv(output_dir / f"scaling_test_{timestamp}.csv", index=False)
    
    # Tạo biểu đồ so sánh
    plt.figure(figsize=(12, 10))
    
    # Execution time comparison
    plt.subplot(2, 1, 1)
    for algo in algorithms.keys():
        algo_results = df_results[df_results['algorithm'] == algo]
        plt.plot(algo_results['size'], algo_results['execution_time'], marker='o', label=algo)
    
    plt.title('Thời gian thực thi theo kích thước đồ thị')
    plt.xlabel('Số lượng đỉnh')
    plt.ylabel('Thời gian (giây)')
    plt.grid(True)
    plt.legend()
    
    # Modularity comparison
    plt.subplot(2, 1, 2)
    for algo in algorithms.keys():
        algo_results = df_results[df_results['algorithm'] == algo]
        plt.plot(algo_results['size'], algo_results['modularity'], marker='o', label=algo)
    
    plt.title('Modularity theo kích thước đồ thị')
    plt.xlabel('Số lượng đỉnh')
    plt.ylabel('Modularity')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f"scaling_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    print(f"Kết quả đã được lưu vào thư mục {output_dir}")
    
    return df_results

def main():
    parser = argparse.ArgumentParser(description='Chạy kiểm tra đánh giá thuật toán phát hiện cộng đồng')
    
    # Tham số đầu vào
    parser.add_argument('--test_type', type=str, choices=['comprehensive', 'scaling'], default='comprehensive',
                        help='Loại kiểm tra (mặc định: comprehensive)')
    
    parser.add_argument('--output_dir', type=str, default="test_results",
                        help='Thư mục lưu kết quả (mặc định: test_results)')
    
    args = parser.parse_args()
    
    # Chạy kiểm tra
    if args.test_type == 'comprehensive':
        run_comprehensive_test(output_dir=args.output_dir)
    else:
        run_scaling_test(output_dir=args.output_dir)

if __name__ == "__main__":
    main() 