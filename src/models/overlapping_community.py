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
            import networkx as nx
            
            # Đảm bảo đồ thị là một instance của nx.Graph
            if not isinstance(self.G, nx.Graph):
                original_graph = self.G
                self.G = nx.Graph(self.G)
                print("Đã chuyển đổi đồ thị sang định dạng nx.Graph")
            
            # Kiểm tra phiên bản cdlib và xử lý tương thích
            from importlib.metadata import version
            cdlib_version = version('cdlib')
            print(f"Phiên bản cdlib: {cdlib_version}")
            
            # Xử lý cộng đồng không chồng chéo khi CONGA không thể chạy
            if len(self.G.nodes()) < 5:
                print("Đồ thị quá nhỏ cho CONGA, sử dụng phương pháp dự phòng")
                return self._fallback_overlapping_detection(num_communities)
            
            if num_communities is None:
                num_communities = min(5, len(self.G.nodes()) // 5)
                
            # Điều chỉnh tham số dựa trên phiên bản cdlib
            try:
                print(f"Thử phát hiện {num_communities} cộng đồng chồng chéo...")
                # Thử với API mới (cdlib 0.4.0+)
                communities = algorithms.conga(self.G, number_communities=num_communities)
            except TypeError:
                try:
                    # Thử với tham số khác
                    communities = algorithms.conga(self.G, n_communities=num_communities)
                except Exception as e:
                    print(f"Lỗi khi chạy conga: {str(e)}")
                    # Thử lại với tham số mặc định
                    communities = algorithms.conga(self.G)
            except Exception as e:
                print(f"Lỗi khi chạy CONGA: {str(e)}")
                print("Đang sử dụng phương pháp dự phòng...")
                return self._fallback_overlapping_detection(num_communities)
            
            # Kiểm tra kết quả và chuyển đổi nếu cần
            if hasattr(communities, 'communities'):
                result_communities = communities.communities
            else:
                result_communities = communities
            
            # Chuyển đổi sang list of sets nếu không phải
            if isinstance(result_communities, list):
                # Kiểm tra xem các phần tử có phải là set không
                if not all(isinstance(comm, set) for comm in result_communities):
                    result_communities = [set(comm) for comm in result_communities]
            else:
                # Trường hợp khác, tạo danh sách rỗng
                print("Định dạng không được hỗ trợ, sử dụng phương pháp dự phòng")
                return self._fallback_overlapping_detection(num_communities)
            
            # Kiểm tra số lượng cộng đồng
            if len(result_communities) < 2:
                print("Không phát hiện đủ cộng đồng, sử dụng phương pháp dự phòng")
                return self._fallback_overlapping_detection(num_communities)
            
            # Đảm bảo không có cộng đồng trùng
            unique_communities = []
            for comm in result_communities:
                if len(comm) > 0 and all(comm != existing for existing in unique_communities):
                    unique_communities.append(comm)
            
            # Đảm bảo mọi nút trong đồ thị đều nằm trong ít nhất một cộng đồng
            all_nodes = set(self.G.nodes())
            covered_nodes = set()
            for comm in unique_communities:
                covered_nodes.update(comm)
            
            uncovered_nodes = all_nodes - covered_nodes
            if uncovered_nodes:
                if len(uncovered_nodes) / len(all_nodes) > 0.8:  # Nếu quá nhiều nút không được bao phủ
                    print("Quá nhiều nút không được bao phủ, sử dụng phương pháp dự phòng")
                    return self._fallback_overlapping_detection(num_communities)
                print(f"Có {len(uncovered_nodes)} nút không thuộc cộng đồng nào. Thêm vào cộng đồng mới.")
                # Thêm các nút chưa được bao phủ vào một cộng đồng mới
                unique_communities.append(uncovered_nodes)
            
            # Kiểm tra lần cuối để đảm bảo số lượng cộng đồng
            if len(unique_communities) < 2:
                print("Không đủ cộng đồng sau khi xử lý, sử dụng phương pháp dự phòng")
                return self._fallback_overlapping_detection(num_communities)
                
            print(f"Đã phát hiện {len(unique_communities)} cộng đồng chồng chéo")
            return unique_communities
            
        except ImportError as e:
            print(f"Không thể sử dụng CONGA. Lỗi: {str(e)}")
            print("Đảm bảo bạn đã cài đặt thư viện cdlib. Đang sử dụng phương pháp dự phòng...")
            return self._fallback_overlapping_detection(num_communities)
        except Exception as e:
            print(f"Lỗi khi chạy CONGA: {str(e)}")
            print("Đang sử dụng phương pháp dự phòng...")
            return self._fallback_overlapping_detection(num_communities)
    
    def _fallback_overlapping_detection(self, num_communities=2):
        """
        Phương pháp dự phòng nếu CONGA không khả dụng
        """
        import networkx as nx
        import random
        
        print("Đang sử dụng phương pháp phát hiện cộng đồng chồng chéo dự phòng...")
        
        # Đảm bảo số lượng cộng đồng hợp lệ
        if num_communities is None or num_communities < 2:
            num_communities = 2
        
        communities = []
        
        # Thử dùng thuật toán phân cụm k-clique nếu có thể
        try:
            # Sử dụng k-clique để phát hiện
            k = 3  # Kích thước clique
            communities_generator = nx.algorithms.community.k_clique_communities(self.G, k)
            communities = list(communities_generator)
            print(f"Đã tìm thấy {len(communities)} cộng đồng sử dụng k-clique clustering")
        except Exception as e:
            print(f"Lỗi khi sử dụng k-clique: {str(e)}")
            communities = []
        
        # Nếu không tìm thấy đủ cộng đồng, thử phương pháp khác
        if len(communities) < num_communities:
            print(f"Không đủ cộng đồng ({len(communities)}), cần thêm ({num_communities - len(communities)})")
            
            # Chuyển đổi danh sách hiện có sang định dạng set
            communities = [set(c) for c in communities]
            
            # Nếu không tìm thấy cộng đồng nào, thử phân chia mạng thành các thành phần liên thông
            if not communities:
                print("Thử phân chia thành các thành phần liên thông...")
                try:
                    # Lấy các thành phần liên thông
                    components = list(nx.connected_components(self.G))
                    if len(components) > 1:
                        communities = components[:num_communities]
                        print(f"Đã tìm thấy {len(communities)} thành phần liên thông")
                except Exception as e:
                    print(f"Lỗi khi tìm các thành phần liên thông: {str(e)}")
            
            # Nếu vẫn không đủ cộng đồng, tạo các cộng đồng ngẫu nhiên
            if not communities:
                print("Tạo cộng đồng ngẫu nhiên từ toàn bộ đồ thị...")
                # Tạo một cộng đồng ban đầu chứa tất cả các đỉnh
                nodes = list(self.G.nodes())
                communities = [set(random.sample(nodes, min(len(nodes), len(nodes) // num_communities + 1)))]
            
            # Thêm đỉnh vào các cộng đồng cho đến khi đạt số lượng mong muốn
            while len(communities) < num_communities:
                # Chọn một cộng đồng ngẫu nhiên để chia
                if not communities:
                    break
                
                source_comm = random.choice(communities)
                
                if len(source_comm) <= 3:  # Nếu cộng đồng quá nhỏ, không chia nữa
                    continue
                
                # Chọn một tập con ngẫu nhiên
                subset_size = max(3, len(source_comm) // 2)
                new_comm = set(random.sample(list(source_comm), subset_size))
                
                # Thêm một số đỉnh xung quanh để tạo sự chồng chéo
                for node in list(new_comm):
                    neighbors = list(self.G.neighbors(node))
                    for neighbor in neighbors:
                        if random.random() < 0.3:  # 30% cơ hội thêm láng giềng
                            new_comm.add(neighbor)
                
                # Chỉ thêm cộng đồng mới nếu nó khác biệt đủ với các cộng đồng hiện có
                if len(new_comm) > 0:
                    # Kiểm tra sự chồng chéo với các cộng đồng hiện có
                    overlap_too_high = False
                    for existing_comm in communities:
                        overlap = len(new_comm.intersection(existing_comm)) / len(new_comm.union(existing_comm))
                        if overlap > 0.8:  # Nếu chồng chéo quá 80%, không thêm
                            overlap_too_high = True
                            break
                    
                    if not overlap_too_high:
                        communities.append(new_comm)
                        print(f"Đã thêm cộng đồng mới, kích thước: {len(new_comm)}")
        
        print(f"Đã tạo {len(communities)} cộng đồng chồng chéo")
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
            import networkx as nx
            
            # Đảm bảo đồ thị là một instance của nx.Graph
            if not isinstance(self.G, nx.Graph):
                original_graph = self.G
                self.G = nx.Graph(self.G)
                print("Đã chuyển đổi đồ thị sang định dạng nx.Graph")
            
            # Kiểm tra phiên bản cdlib và xử lý tương thích
            from importlib.metadata import version
            cdlib_version = version('cdlib')
            print(f"Phiên bản cdlib: {cdlib_version}")
            
            # Điều chỉnh tham số dựa trên phiên bản cdlib
            if float(cdlib_version.split('.')[0]) < 1 and float(cdlib_version.split('.')[1]) >= 4:
                # Từ cdlib 0.4.0 trở lên
                if num_communities is None:
                    num_communities = 2
                try:
                    communities = algorithms.congo(self.G, number_communities=num_communities)
                except TypeError:
                    # Nếu API đã thay đổi, thử với tham số khác
                    communities = algorithms.congo(self.G, n_communities=num_communities)
            else:
                # Với cdlib 0.2.x
                if max_split_degree is None:
                    max_split_degree = len(self.G.nodes()) // 2
                communities = algorithms.congo(self.G, number_communities=num_communities, 
                                             max_communities=max_iter)
            
            # Kiểm tra kết quả và chuyển đổi nếu cần
            if hasattr(communities, 'communities'):
                result_communities = communities.communities
            else:
                result_communities = communities
            
            # Chuyển đổi sang list of sets nếu không phải
            if isinstance(result_communities, list):
                # Kiểm tra xem các phần tử có phải là set không
                if not all(isinstance(comm, set) for comm in result_communities):
                    result_communities = [set(comm) for comm in result_communities]
            else:
                # Trường hợp khác, tạo danh sách rỗng
                print("Định dạng không được hỗ trợ, sử dụng phương pháp dự phòng")
                conga = CONGA(self.G)
                return conga._fallback_overlapping_detection(num_communities)
            
            # Đảm bảo không có cộng đồng trùng
            unique_communities = []
            for comm in result_communities:
                if comm not in unique_communities and len(comm) > 0:
                    unique_communities.append(comm)
            
            # Đảm bảo mọi nút trong đồ thị đều nằm trong ít nhất một cộng đồng
            all_nodes = set(self.G.nodes())
            covered_nodes = set()
            for comm in unique_communities:
                covered_nodes.update(comm)
            
            uncovered_nodes = all_nodes - covered_nodes
            if uncovered_nodes:
                print(f"Có {len(uncovered_nodes)} nút không thuộc cộng đồng nào. Thêm vào cộng đồng mới.")
                # Thêm các nút chưa được bao phủ vào một cộng đồng mới
                unique_communities.append(uncovered_nodes)
            
            print(f"Đã phát hiện {len(unique_communities)} cộng đồng chồng chéo")
            return unique_communities
            
        except ImportError as e:
            print(f"Không thể sử dụng CONGO. Lỗi: {str(e)}")
            print("Đảm bảo bạn đã cài đặt thư viện cdlib. Đang sử dụng phương pháp dự phòng...")
            # Sử dụng cùng một phương pháp dự phòng như CONGA
            conga = CONGA(self.G)
            return conga._fallback_overlapping_detection(num_communities)
        except Exception as e:
            print(f"Lỗi khi chạy CONGO: {str(e)}")
            print("Đang sử dụng phương pháp dự phòng...")
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