import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import random
import plotly.graph_objects as go
from matplotlib.colors import to_rgba

def set_node_community(G, communities):
    """
    Thêm thuộc tính cộng đồng vào các đỉnh của đồ thị
    
    Args:
        G: networkx.Graph
        communities: dict mapping node -> community_id hoặc list of sets
    """
    # Chuyển đổi communities thành dict nếu là list of sets
    if isinstance(communities, list):
        community_dict = {}
        for i, comm in enumerate(communities):
            for node in comm:
                if node in community_dict:
                    if isinstance(community_dict[node], set):
                        community_dict[node].add(i)
                    else:
                        community_dict[node] = {community_dict[node], i}
                else:
                    community_dict[node] = i
        communities = community_dict
    
    # Thêm thuộc tính vào đồ thị
    for node in G.nodes():
        if node in communities:
            G.nodes[node]['community'] = communities[node]
        else:
            G.nodes[node]['community'] = -1  # Gán cho cộng đồng mặc định

def get_color_dict(num_colors):
    """
    Tạo từ điển màu sắc
    
    Args:
        num_colors: Số lượng màu cần tạo
        
    Returns:
        dict: Từ điển ánh xạ community_id -> màu sắc
    """
    colors = {}
    
    # Sử dụng colormap tab20 cho 20 màu đầu tiên
    if num_colors <= 20:
        cmap = cm.get_cmap('tab20', num_colors)
        for i in range(num_colors):
            colors[i] = to_rgba(cmap(i))
    else:
        # Nếu cần nhiều hơn 20 màu, tạo màu ngẫu nhiên
        used_colors = set()
        cmap = cm.get_cmap('tab20', 20)
        
        # Sử dụng 20 màu từ tab20 trước
        for i in range(min(20, num_colors)):
            color = tuple(cmap(i))
            colors[i] = color
            used_colors.add(color)
        
        # Tạo các màu ngẫu nhiên cho phần còn lại
        for i in range(20, num_colors):
            while True:
                color = tuple([random.random() for _ in range(3)] + [1.0])
                # Đảm bảo màu không quá gần với các màu đã sử dụng
                if all(sum((c1 - c2) ** 2 for c1, c2 in zip(color, existing_color)) > 0.1 
                      for existing_color in used_colors):
                    colors[i] = color
                    used_colors.add(color)
                    break
    
    return colors

def visualize_communities(G, communities, title="Community Detection", figsize=(12, 10),
                         node_size=100, edge_alpha=0.2, with_labels=True, 
                         pos=None, save_path=None):
    """
    Trực quan hóa các cộng đồng trong đồ thị
    
    Args:
        G: networkx.Graph
        communities: dict mapping node -> community_id hoặc list of sets
        title: Tiêu đề của đồ thị
        figsize: Kích thước figure
        node_size: Kích thước đỉnh
        edge_alpha: Độ mờ của cạnh
        with_labels: Hiển thị nhãn đỉnh
        pos: Vị trí của các đỉnh
        save_path: Đường dẫn để lưu hình ảnh
    """
    # Xác định vị trí các đỉnh nếu không được cung cấp
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    # Thêm thuộc tính cộng đồng vào đồ thị
    set_node_community(G, communities)
    
    # Xác định số lượng cộng đồng
    if isinstance(communities, dict):
        community_ids = set()
        for node, comm in communities.items():
            if isinstance(comm, (set, frozenset)):
                community_ids.update(comm)
            else:
                community_ids.add(comm)
        num_communities = len(community_ids)
    else:
        num_communities = len(communities)
    
    # Tạo từ điển màu sắc
    color_dict = get_color_dict(num_communities)
    
    # Xác định màu cho mỗi đỉnh
    node_colors = []
    for node in G.nodes():
        if node in G.nodes and 'community' in G.nodes[node]:
            comm = G.nodes[node]['community']
            if isinstance(comm, (set, frozenset)):
                # Đối với cộng đồng chồng chéo, lấy một màu ngẫu nhiên từ các cộng đồng
                comm_list = list(comm)
                node_colors.append(color_dict[random.choice(comm_list)])
            else:
                node_colors.append(color_dict.get(comm, (0.5, 0.5, 0.5, 1.0)))
        else:
            node_colors.append((0.5, 0.5, 0.5, 1.0))  # Màu xám cho các đỉnh không thuộc cộng đồng nào
    
    # Tạo figure
    plt.figure(figsize=figsize)
    
    # Vẽ đồ thị
    nx.draw_networkx_edges(G, pos, alpha=edge_alpha)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size)
    
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_overlapping_communities(G, communities, title="Overlapping Communities", 
                                    figsize=(12, 10), save_path=None):
    """
    Trực quan hóa các cộng đồng chồng chéo trong đồ thị
    
    Args:
        G: networkx.Graph
        communities: list of sets, mỗi set chứa các đỉnh trong một cộng đồng
        title: Tiêu đề của đồ thị
        figsize: Kích thước figure
        save_path: Đường dẫn để lưu hình ảnh
    """
    # Xác định vị trí các đỉnh
    pos = nx.spring_layout(G, seed=42)
    
    # Tạo color map
    color_dict = get_color_dict(len(communities))
    
    # Tạo figure
    plt.figure(figsize=figsize)
    
    # Vẽ tất cả các cạnh với màu xám nhạt
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')
    
    # Vẽ tất cả các đỉnh với màu xám nhạt
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=50)
    
    # Vẽ mỗi cộng đồng với một màu riêng
    for i, community in enumerate(communities):
        subgraph = G.subgraph(community)
        nx.draw_networkx_nodes(subgraph, pos, node_color=[color_dict[i]] * len(subgraph), 
                              node_size=200, alpha=0.6)
    
    # Vẽ nhãn
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_interactive(G, communities, title="Interactive Community Visualization", 
                         is_overlapping=False, open_browser=True):
    """
    Tạo biểu đồ tương tác sử dụng Plotly
    
    Args:
        G: networkx.Graph
        communities: dict mapping node -> community_id hoặc list of sets
        title: Tiêu đề của đồ thị
        is_overlapping: Cộng đồng có chồng chéo không
        open_browser: Mở trình duyệt sau khi tạo
        
    Returns:
        plotly.graph_objects.Figure
    """
    # Xác định vị trí các đỉnh
    pos = nx.spring_layout(G, seed=42)
    
    # Chuyển đổi communities thành định dạng phù hợp
    if is_overlapping and isinstance(communities, list):
        community_dict = {}
        for i, comm in enumerate(communities):
            for node in comm:
                if node in community_dict:
                    if isinstance(community_dict[node], set):
                        community_dict[node].add(i)
                    else:
                        community_dict[node] = {community_dict[node], i}
                else:
                    community_dict[node] = i
        communities = community_dict
    
    # Thêm thuộc tính cộng đồng vào đồ thị
    set_node_community(G, communities)
    
    # Xác định số lượng cộng đồng
    if isinstance(communities, dict):
        community_ids = set()
        for node, comm in communities.items():
            if isinstance(comm, (set, frozenset)):
                community_ids.update(comm)
            else:
                community_ids.add(comm)
        num_communities = len(community_ids)
    else:
        num_communities = len(communities)
    
    # Tạo từ điển màu sắc
    color_dict = get_color_dict(num_communities)
    
    # Chuẩn bị dữ liệu cho Plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Tạo trace cho cạnh
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Chuẩn bị dữ liệu cho đỉnh
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Xác định màu sắc dựa trên cộng đồng
        if 'community' in G.nodes[node]:
            comm = G.nodes[node]['community']
            if isinstance(comm, (set, frozenset)):
                # Đối với cộng đồng chồng chéo, hiển thị tất cả các ID cộng đồng
                color = color_dict[list(comm)[0]]  # Lấy màu của cộng đồng đầu tiên
                comm_str = ', '.join(str(c) for c in sorted(comm))
                node_text.append(f"Node: {node}<br>Communities: {comm_str}")
            else:
                color = color_dict.get(comm, (0.5, 0.5, 0.5, 1.0))
                node_text.append(f"Node: {node}<br>Community: {comm}")
        else:
            color = (0.5, 0.5, 0.5, 1.0)  # Màu xám cho các đỉnh không thuộc cộng đồng nào
            node_text.append(f"Node: {node}")
        
        # Chuyển đổi màu thành định dạng rgb cho Plotly
        rgba_color = tuple(int(c * 255) for c in color[:3])
        node_colors.append(f'rgba{rgba_color + (color[3],)}')
    
    # Tạo trace cho đỉnh
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=10,
            color=node_colors,
            line_width=2
        )
    )
    
    # Tạo figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=title,
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    if open_browser:
        fig.show()
    
    return fig 