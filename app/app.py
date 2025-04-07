import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go

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
from src.visualization import visualize_communities, visualize_overlapping_communities, visualize_interactive

# Thiết lập tiêu đề trang web
st.set_page_config(
    page_title="Phát hiện cộng đồng trong mạng xã hội",
    page_icon="🔍",
    layout="wide"
)

# Tiêu đề
st.title("Phát hiện cộng đồng trong mạng xã hội")
st.markdown("""
    Demo ứng dụng các thuật toán phát hiện cộng đồng trong mạng xã hội.
    Chọn bộ dữ liệu, thuật toán và các tham số để xem kết quả phát hiện cộng đồng.
""")

# Thanh sidebar
st.sidebar.header("Cài đặt")

# Tải dữ liệu
@st.cache_resource
def load_dataset(dataset_name, n=100, k=5, overlap_size=10, p_in=0.3, p_out=0.05):
    """
    Tải bộ dữ liệu
    
    Args:
        dataset_name: Tên bộ dữ liệu
        n, k, overlap_size, p_in, p_out: Tham số cho bộ dữ liệu tổng hợp
        
    Returns:
        G: networkx.Graph
        true_communities: Danh sách các cộng đồng thực sự (nếu có)
    """
    dataset = GraphDataset()
    
    if dataset_name == "karate_club":
        G = dataset.load_karate_club()
        true_communities = None
    elif dataset_name == "dolphins":
        G = dataset.load_dolphins()
        true_communities = None
    elif dataset_name == "football":
        G, true_communities = dataset.load_football()
    elif dataset_name == "synthetic":
        G, true_communities = dataset.generate_synthetic_data(n=n, k=k)
    elif dataset_name == "synthetic_overlapping":
        G, true_communities = dataset.generate_overlapping_communities(n=n, k=k, overlap_size=overlap_size, p_in=p_in, p_out=p_out)
    else:
        st.error(f"Bộ dữ liệu {dataset_name} không được hỗ trợ")
        return None, None
    
    return G, true_communities

# Lấy danh sách thuật toán
def get_disjoint_algorithms():
    return {
        "Girvan-Newman": GirvanNewman,
        "Louvain": Louvain,
        "Spectral Clustering": SpectralCommunity,
        "K-means": KMeansCommunity,
        "Hierarchical Clustering": HierarchicalCommunity,
        "Label Propagation": LabelPropagation,
        "Infomap": InfomapCommunity
    }

def get_overlapping_algorithms():
    return {
        "CONGA": CONGA,
        "CONGO": CONGO,
        "COPRA": COPRA,
        "BIGCLAM": BIGCLAM
    }

# Chọn bộ dữ liệu
dataset_options = {
    "karate_club": "Zachary's Karate Club",
    "dolphins": "Dolphins Social Network",
    "football": "American College Football",
    "synthetic": "Dữ liệu tổng hợp",
    "synthetic_overlapping": "Dữ liệu tổng hợp (cộng đồng chồng chéo)"
}

dataset_name = st.sidebar.selectbox(
    "Chọn bộ dữ liệu:",
    list(dataset_options.keys()),
    format_func=lambda x: dataset_options[x]
)

# Tham số cho bộ dữ liệu tổng hợp
if dataset_name in ["synthetic", "synthetic_overlapping"]:
    n = st.sidebar.slider("Số lượng đỉnh (n):", min_value=20, max_value=500, value=100, step=10)
    k = st.sidebar.slider("Số lượng cộng đồng (k):", min_value=2, max_value=10, value=5)
    p_in = st.sidebar.slider("Xác suất liên kết trong cộng đồng (p_in):", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
    p_out = st.sidebar.slider("Xác suất liên kết ngoài cộng đồng (p_out):", min_value=0.0, max_value=0.3, value=0.05, step=0.01)
    
    if dataset_name == "synthetic_overlapping":
        overlap_size = st.sidebar.slider("Kích thước phần chồng chéo:", min_value=1, max_value=50, value=10)
    else:
        overlap_size = 0
else:
    n, k, p_in, p_out, overlap_size = 100, 5, 0.3, 0.05, 10

# Tải dữ liệu
G, true_communities = load_dataset(dataset_name, n, k, overlap_size, p_in, p_out)

# Hiển thị thông tin về đồ thị
if G:
    st.sidebar.info(f"Đồ thị có {G.number_of_nodes()} đỉnh và {G.number_of_edges()} cạnh")

# Chọn loại thuật toán
algorithm_type = st.sidebar.radio(
    "Loại thuật toán:",
    ["Cộng đồng không chồng chéo", "Cộng đồng chồng chéo"]
)

# Chọn thuật toán cụ thể
if algorithm_type == "Cộng đồng không chồng chéo":
    algorithms = get_disjoint_algorithms()
    is_overlapping = False
else:
    algorithms = get_overlapping_algorithms()
    is_overlapping = True

algorithm_name = st.sidebar.selectbox(
    "Chọn thuật toán:",
    list(algorithms.keys())
)

# Tham số cho thuật toán
st.sidebar.subheader("Tham số thuật toán")

algorithm_params = {}
if algorithm_name == "Girvan-Newman":
    num_communities = st.sidebar.slider("Số lượng cộng đồng:", min_value=2, max_value=10, value=5)
    max_iter = st.sidebar.slider("Số lần lặp tối đa:", min_value=5, max_value=100, value=10)
    algorithm_params = {"num_communities": num_communities, "max_iter": max_iter}
elif algorithm_name in ["Spectral Clustering", "K-means", "Hierarchical Clustering"]:
    n_clusters = st.sidebar.slider("Số lượng cộng đồng:", min_value=2, max_value=10, value=5)
    algorithm_params = {"n_clusters": n_clusters}
elif algorithm_name == "COPRA":
    v = st.sidebar.slider("Số lượng cộng đồng tối đa mỗi đỉnh (v):", min_value=1, max_value=5, value=2)
    max_iter = st.sidebar.slider("Số lần lặp tối đa:", min_value=5, max_value=100, value=30)
    algorithm_params = {"v": v, "max_iter": max_iter}
elif algorithm_name in ["CONGA", "CONGO"]:
    num_communities = st.sidebar.slider("Số lượng cộng đồng:", min_value=2, max_value=10, value=5)
    algorithm_params = {"num_communities": num_communities}
elif algorithm_name == "BIGCLAM":
    num_communities = st.sidebar.slider("Số lượng cộng đồng:", min_value=2, max_value=10, value=5)
    algorithm_params = {"num_communities": num_communities}

# Nút thực thi
execute_button = st.sidebar.button("Thực thi thuật toán")

if G is not None and execute_button:
    # Tạo hai cột
    col1, col2 = st.columns([3, 2])
    
    # Hiển thị đồ thị gốc
    with col1:
        st.subheader("Đồ thị gốc")
        
        # Tạo đồ thị tương tác với Plotly
        fig = visualize_interactive(G, {}, title="Đồ thị gốc", open_browser=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Chạy thuật toán
    with st.spinner(f"Đang chạy thuật toán {algorithm_name}..."):
        start_time = time.time()
        
        # Khởi tạo và chạy thuật toán
        algorithm_class = algorithms[algorithm_name]
        algorithm = algorithm_class(G)
        communities = algorithm.detect(**algorithm_params)
        
        end_time = time.time()
        execution_time = end_time - start_time
    
    # Hiển thị kết quả
    with col2:
        st.subheader("Thông tin kết quả")
        
        # Tính các chỉ số đánh giá
        mod = modularity(G, communities)
        cond = conductance(G, communities)
        
        # Hiển thị thông tin
        st.write(f"**Thuật toán:** {algorithm_name}")
        
        if isinstance(communities, dict):
            num_communities = len(set(communities.values()))
        else:
            num_communities = len(communities)
            
        st.write(f"**Số lượng cộng đồng phát hiện được:** {num_communities}")
        st.write(f"**Modularity:** {mod:.4f}")
        st.write(f"**Conductance (trung bình):** {cond['average']:.4f}")
        st.write(f"**Thời gian thực thi:** {execution_time:.4f} giây")
        
        # Tính NMI và ARI nếu có ground truth
        if true_communities is not None:
            if is_overlapping:
                if isinstance(true_communities, dict):
                    # Chuyển từ dict sang list of sets
                    true_comm_dict = {}
                    for node, comm_id in true_communities.items():
                        if comm_id not in true_comm_dict:
                            true_comm_dict[comm_id] = set()
                        true_comm_dict[comm_id].add(node)
                    true_comm_list = list(true_comm_dict.values())
                else:
                    true_comm_list = true_communities
                
                onmi_score = overlapping_nmi(true_comm_list, communities)
                omega_score = omega_index(true_comm_list, communities)
                
                st.write(f"**Overlapping NMI:** {onmi_score:.4f}" if onmi_score else "**Overlapping NMI:** N/A")
                st.write(f"**Omega Index:** {omega_score:.4f}" if omega_score else "**Omega Index:** N/A")
            else:
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
                    
                    st.write(f"**NMI:** {nmi_score:.4f}")
                    st.write(f"**ARI:** {ari_score:.4f}")
    
    # Hiển thị đồ thị với cộng đồng
    with col1:
        st.subheader("Kết quả phát hiện cộng đồng")
        
        # Tạo đồ thị tương tác với Plotly
        fig = visualize_interactive(G, communities, title=f"Kết quả {algorithm_name}", 
                                  is_overlapping=is_overlapping, open_browser=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị phân phối kích thước cộng đồng
        st.subheader("Phân phối kích thước cộng đồng")
        
        if isinstance(communities, dict):
            # Tính kích thước mỗi cộng đồng
            community_sizes = {}
            for node, comm_id in communities.items():
                if isinstance(comm_id, (set, frozenset)):
                    for c in comm_id:
                        if c not in community_sizes:
                            community_sizes[c] = 0
                        community_sizes[c] += 1
                else:
                    if comm_id not in community_sizes:
                        community_sizes[comm_id] = 0
                    community_sizes[comm_id] += 1
        else:
            # communities là list of sets
            community_sizes = {i: len(comm) for i, comm in enumerate(communities)}
        
        # Tạo dataframe
        df_sizes = pd.DataFrame({
            'Cộng đồng': list(community_sizes.keys()),
            'Kích thước': list(community_sizes.values())
        })
        
        # Vẽ biểu đồ
        fig = px.bar(df_sizes, x='Cộng đồng', y='Kích thước', 
                    title='Kích thước các cộng đồng')
        st.plotly_chart(fig, use_container_width=True)
        
else:
    # Hiển thị hướng dẫn
    st.info("Chọn bộ dữ liệu và thuật toán từ thanh bên, sau đó nhấn nút 'Thực thi thuật toán' để xem kết quả.")
    
    # Hiển thị thông tin về các thuật toán
    st.subheader("Thông tin về các thuật toán")
    
    st.markdown("""
    ### Thuật toán phát hiện cộng đồng không chồng chéo
    
    1. **Girvan-Newman**: Thuật toán dựa trên phân chia mạng bằng cách lặp đi lặp lại việc loại bỏ các cạnh có độ trung gian cao nhất.
    
    2. **Louvain**: Thuật toán tối ưu hóa modularity thông qua phương pháp tham lam.
    
    3. **Spectral Clustering**: Sử dụng vector riêng của ma trận Laplacian của đồ thị để phân cụm.
    
    4. **K-means**: Áp dụng thuật toán K-means lên ma trận kề của đồ thị.
    
    5. **Hierarchical Clustering**: Phân cụm phân cấp dựa trên khoảng cách giữa các đỉnh.
    
    6. **Label Propagation**: Phát hiện cộng đồng bằng cách lan truyền nhãn giữa các đỉnh.
    
    7. **Infomap**: Sử dụng lý thuyết thông tin để phát hiện cộng đồng tốt nhất.
    
    ### Thuật toán phát hiện cộng đồng chồng chéo
    
    1. **CONGA (Cluster-Overlap Newman Girvan Algorithm)**: Mở rộng thuật toán Girvan-Newman để phát hiện cộng đồng chồng chéo.
    
    2. **CONGO (CONGA Optimized)**: Phiên bản tối ưu hóa của CONGA, sử dụng trung gian cục bộ thay vì toàn cục.
    
    3. **COPRA (Community Overlap PRopagation Algorithm)**: Mở rộng thuật toán lan truyền nhãn cho cộng đồng chồng chéo.
    
    4. **BIGCLAM (Cluster Affiliation Model for Big Networks)**: Sử dụng mô hình xác suất để phát hiện cộng đồng chồng chéo.
    """)
    
    # Hiển thị thông tin về độ đo
    st.subheader("Các độ đo đánh giá")
    
    st.markdown("""
    ### Độ đo đánh giá bên trong (Internal)
    
    1. **Modularity**: Đo lường chất lượng của việc chia mạng thành các cộng đồng. Giá trị cao hơn cho thấy cấu trúc cộng đồng tốt hơn.
    
    2. **Conductance**: Đo lường tỷ lệ giữa số cạnh đi ra ngoài cộng đồng so với tổng số cạnh. Giá trị thấp hơn tốt hơn.
    
    ### Độ đo đánh giá bên ngoài (External)
    
    1. **NMI (Normalized Mutual Information)**: Đo lường mức độ tương đồng giữa hai phân hoạch. Giá trị từ 0 đến 1, 1 là tốt nhất.
    
    2. **ARI (Adjusted Rand Index)**: Đo lường sự tương đồng giữa hai phân hoạch, đã điều chỉnh cho ngẫu nhiên. Giá trị từ -1 đến 1, 1 là tốt nhất.
    
    3. **Overlapping NMI**: Phiên bản của NMI cho cộng đồng chồng chéo.
    
    4. **Omega Index**: Mở rộng của ARI cho cộng đồng chồng chéo.
    """)

# Footer
st.markdown("---")
st.markdown("Phát hiện cộng đồng trong mạng xã hội - Demo AI") 