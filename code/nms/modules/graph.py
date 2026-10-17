import numpy as np
import heapq
import nms.utils.utils as utils
import pickle as pkl
from pathlib import Path
from nms.rotation.rotation_numpy import ortho6d_to_rotmat, rotmat_to_aaxis, rotmat_to_quat, quat_closest, aaxis_to_rotmat
from nms.training.process import recon_pos_from_disp
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

class ClusterTransitionGraph:
    def __init__(
        self,
        vqvae_dir,
        graph_dir,
        weights: dict = None,
        load_precomputed: bool = False,
        load_dist_cache: bool = True,
        control: bool = False
    ):
        self.load_adj(graph_dir)
        self.load_datarep(vqvae_dir)
        self.load_codebook(vqvae_dir)
        self.init_weights(weights) # will not be in use if load_precomputed == True
        self.init_dist(graph_dir, load_dist_cache)

        self.load_precomputed = load_precomputed
        if load_precomputed:
            # load precomputed edges and heu values
            self.load_cost(graph_dir)
        
        self.control = control
        if control:
            # precompute orientations
            root_orient = self.codebook_decoded[..., self.feat_start_idx["root_orient"]:self.feat_end_idx["root_orient"]]
            root_orient = ortho6d_to_rotmat(root_orient)
            self.decoded_root_orient = root_orient # [n_cluster, 16, 3, 3]
            
            # precompute root translational displacement
            root_trans_disp = self.codebook_decoded[..., self.feat_start_idx["root_trans_disp"]:self.feat_end_idx["root_trans_disp"]]
            root_trans_disp = np.expand_dims(root_trans_disp, axis=-2)
            root_trans_disp = recon_pos_from_disp(root_trans_disp) # [n_cluster, 17, 1, 3]
            self.decoded_accum_root_trans_disp = root_trans_disp[:, -2] # [n_cluster, 1, 3] keep only the last frame

            
    def load_adj(self, graph_dir):
        adj = sp.load_npz(f"{graph_dir}/valid_trans_bool.npz")
        n_components, labels = connected_components(csgraph=adj, directed=True, connection='strong')
        components = [np.where(labels == k)[0] for k in range(n_components)]
        components.sort(key=len, reverse=True)
        
        # find valid nodes(== clusters)
        largest_comp_idx = np.bincount(labels).argmax()
        self.largest_comp_nodes = np.where(labels == largest_comp_idx)[0] # save nodes in largest components as valid nodes
        self.not_in_largest = np.where(labels != largest_comp_idx)[0]
        self.adj = adj.toarray()
    
    def load_datarep(self, vqvae_dir, datarep_name="vqvae_datarep"):
        self.datarep = utils.load_config(vqvae_dir, datarep_name, to_dict=True)
        _, feat_start_idx, feat_end_idx, _, _ = utils.get_feature_dims(self.datarep)
        self.feat_start_idx = feat_start_idx
        self.feat_end_idx = feat_end_idx

    def load_codebook(self, vqvae_dir):
        codebook_dir = f"{vqvae_dir}/codebook"
        self.codebook_decoded = np.load(f"{codebook_dir}/codebook_decoded.npy")
        self.codebook = np.load(f"{codebook_dir}/codebook.npy")
        self.n_node = self.codebook.shape[0]
    
    def init_weights(self, weights: dict = None):
        # if no weight is given, init to 1.0
        if weights is None:
            keys = list(self.datarep["motion"].keys())
            self.weights = {key: 1.0 for key in keys}
            self.weights["code"] = 1.0 # init for cosine distance between codes
        else:
            self.weights = weights
    
    def init_dist(self, graph_dir, load_dist_cache=True):
        self.graph_dir = graph_dir
        dist_path = Path(f"{self.graph_dir}/dist.pkl")
        if not load_dist_cache or not dist_path.exists():
            dist = self.compute_dist_pairwise() # batch process all distance values
            with open(f"{self.graph_dir}/dist.pkl", "wb") as f:
                pkl.dump(dist, f)
            print(f"Distance values newly computed and saved at: {self.graph_dir}/dist.pkl")
        else:
            with open(f"{self.graph_dir}/dist.pkl", "rb") as f:
                dist = pkl.load(f)
            print(f"Distance values loaded from: {self.graph_dir}/dist.pkl")
        
        self.dist = dist
        self.dist_min, self.dist_max = self.compute_dist_min_max(dist)
        self.dist_scaled_mean, self.dist_scaled_std = self.compute_dist_scaled_mean_std(dist)
    
    def init_graph(self, weights = None):
        assert self.adj is not None, "valid_trans_bool should be given to initialize graph"
        
        if weights is None:
            weights = self.weights
        
        print("Initializing graph...")
        N = self.adj.shape[0]
        
        if self.load_precomputed: # cost is already loaded
            self.graph = np.where(self.adj == 1, self.all_pair_cost, np.inf)
        else:
            graph = np.full((N, N), float("inf"), dtype=float)
            for i in range(N):
                for j in range(N):
                    if self.adj[i][j]:
                        graph[i][j] = self.compute_cost(i, j, weights)
            self.graph = graph
            
        # exclude the nodes not in the biggest component --> will not be searched during A*
        self.graph[self.not_in_largest, :] = np.inf
        self.graph[:, self.not_in_largest] = np.inf
                    
        # keep an adjacency list
        graph_adj_list = [[] for _ in range(N)]
        for i in range(N):
            row = self.graph[i]
            for j, val in enumerate(row):
                if val != float("inf"):
                    graph_adj_list[i].append((j, val))
        self.graph_adj_list = graph_adj_list

    
    '''
    Dist related
    '''
    def compute_dist_pairwise(self):
        assert self.feat_start_idx is not None, "feat_start_idx is not loaded"
        assert self.feat_end_idx is not None, "feat_end_idx is not loaded"
        assert self.codebook is not None, "codebook is not loaded"
        assert self.codebook_decoded is not None, "codebook_decoded is not loaded"
        
        feat_start_idx = self.feat_start_idx
        feat_end_idx = self.feat_end_idx
        
        # compare boundary
        featA_E = self.codebook_decoded[..., -1, :].copy() # last frame
        featB_0 = self.codebook_decoded[..., 0, :].copy() # first frame
        
        # expand for pairwise computation
        featA_E = featA_E[:, np.newaxis, :]
        featB_0 = featB_0[np.newaxis, :, :]
        
        #   - root height
        root_h_A = featA_E[..., feat_start_idx["root_height"]:feat_end_idx["root_height"]] 
        root_h_B = featB_0[..., feat_start_idx["root_height"]:feat_end_idx["root_height"]] 
        diff_root_height = np.linalg.norm(root_h_B - root_h_A, axis=-1)
        
        #   - root orientation
        root_orient_from = featA_E[..., feat_start_idx["root_orient"]:feat_end_idx["root_orient"]]
        root_orient_to = featB_0[..., feat_start_idx["root_orient"]:feat_end_idx["root_orient"]]
        root_orient_from = ortho6d_to_rotmat(root_orient_from)
        root_orient_to = ortho6d_to_rotmat(root_orient_to)
        
        diff_root_orient = root_orient_to @ np.linalg.inv(root_orient_from)
        diff_root_orient = rotmat_to_aaxis(diff_root_orient)
        diff_root_orient = np.linalg.norm(diff_root_orient, axis=-1) + 1e-8
        
        #   - root translational displacement        
        root_trans_disp_A = featA_E[..., feat_start_idx["root_trans_disp"]:feat_end_idx["root_trans_disp"]] 
        root_trans_disp_B = featB_0[..., feat_start_idx["root_trans_disp"]:feat_end_idx["root_trans_disp"]] 
        diff_root_trans_disp = np.linalg.norm(root_trans_disp_B - root_trans_disp_A, axis=-1)
        
        #   - root angular displacement
        root_ang_disp_from = featA_E[..., feat_start_idx["root_ang_disp"]:feat_end_idx["root_ang_disp"]]
        root_ang_disp_to = featB_0[..., feat_start_idx["root_ang_disp"]:feat_end_idx["root_ang_disp"]]
        root_ang_disp_from = ortho6d_to_rotmat(root_ang_disp_from)
        root_ang_disp_to = ortho6d_to_rotmat(root_ang_disp_to)
        
        diff_root_ang_disp = root_ang_disp_to @ np.linalg.inv(root_ang_disp_from)
        diff_root_ang_disp = rotmat_to_aaxis(diff_root_ang_disp)
        diff_root_ang_disp = np.linalg.norm(diff_root_ang_disp, axis=-1) + 1e-8   
        
        #   - joint position (root relative)
        joint_pos_A = featA_E[..., feat_start_idx["joint_pos_root_rel"]:feat_end_idx["joint_pos_root_rel"]]
        joint_pos_B = featB_0[..., feat_start_idx["joint_pos_root_rel"]:feat_end_idx["joint_pos_root_rel"]]
        joint_pos_A = joint_pos_A.reshape([*joint_pos_A.shape[:-1], -1, 3])
        joint_pos_B = joint_pos_B.reshape([*joint_pos_B.shape[:-1], -1, 3])
        
        diff_joint_pos = np.linalg.norm(joint_pos_B - joint_pos_A, axis=-1)
        diff_joint_pos_all = np.mean(diff_joint_pos, axis=-1)
        
        #   - joint_trans_disp (world)
        joint_trans_disp_A = featA_E[..., feat_start_idx["joint_trans_disp"]:feat_end_idx["joint_trans_disp"]]
        joint_trans_disp_B = featB_0[..., feat_start_idx["joint_trans_disp"]:feat_end_idx["joint_trans_disp"]]
        joint_trans_disp_A = joint_trans_disp_A.reshape([*joint_trans_disp_A.shape[:-1], -1, 3])
        joint_trans_disp_B = joint_trans_disp_B.reshape([*joint_trans_disp_B.shape[:-1], -1, 3])
        diff_joint_trans_disp = np.linalg.norm(joint_trans_disp_B - joint_trans_disp_A, axis=-1)
        diff_joint_trans_disp = np.mean(diff_joint_trans_disp, axis=-1)
        
        # compare block-wise cos dist between codes
        cos_sim = utils.compute_pairwise_similarity(self.codebook, self.codebook, distance='cos')
        cos_dist = 1 - cos_sim

        # collect
        dist = {
            "root_height": diff_root_height,
            "root_orient": diff_root_orient,
            "root_trans_disp": diff_root_trans_disp,
            "root_ang_disp": diff_root_ang_disp,
            "joint_pos_root_rel": diff_joint_pos_all,
            "joint_trans_disp": diff_joint_trans_disp,
            "code": cos_dist
        }
        return dist
    
    def compute_dist_min_max(self, dist):
        dist_min = {key: float("inf") for key in dist.keys()}
        dist_max = {key: float("-inf") for key in dist.keys()}
        
        for k in dist.keys():
            dist_min[k] = dist[k].min()
            dist_max[k] = dist[k].max()
        return dist_min, dist_max

    def compute_dist_scaled_mean_std(self, dist):
        assert self.dist_min is not None
        assert self.dist_max is not None
        
        dist_scaled = {}
        dist_scaled_mean = {}
        dist_scaled_std = {}
        for k in dist.keys():
            # scale and apply weight
            dist_scaled[k] = (dist[k] - self.dist_min[k]) / (self.dist_max[k] - self.dist_min[k])
            dist_scaled_mean[k] = dist_scaled[k].mean()
            dist_scaled_std[k] = dist_scaled[k].std()
        
        return dist_scaled_mean, dist_scaled_std
    
    
    '''
    Cost related
    '''
    def compute_cost(
        self,
        cur_idx: int,
        next_idx: int,
        weights: dict
    ):
        cost = []
        if self.load_precomputed:
            return self.all_pair_cost[cur_idx, next_idx]
        
        for k in weights.keys():
            if k == "control": 
                continue
            dist_scaled = (self.dist[k][cur_idx, next_idx] - self.dist_min[k]) / (self.dist_max[k] - self.dist_min[k])
            dist_scaled = dist_scaled / (self.dist_scaled_mean[k] + 1e-8) # divide by mean to match scale
            dist_weighted = dist_scaled * weights[k]
            cost.append(dist_weighted)

        return sum(cost)
    
    def compute_cost_control(
        self,
        cur_idx: int,
        next_idx: int,
        accum_root_trf: np.ndarray,
        control: np.ndarray,
        edge: bool = True
    ):
        if edge: # already precomputed
            cost = self.graph[cur_idx, next_idx]
        else:
            cost = self.compute_cost(cur_idx, next_idx, weights=self.weights)
        
        # accumulate root pos for control
        accum_root_pos = accum_root_trf[..., :3, 3]
        accum_root_pos_exp = np.reshape(accum_root_pos, [1, 1, 1, *accum_root_pos.shape])
        
        root_orient_target = accum_root_trf[..., :3, :3]
        root_orient = self.decoded_root_orient[next_idx]
        anchor = root_orient[..., :1, :, :] # original first root orient as an anchor
        anchor_inv = np.linalg.inv(anchor)
        
        up_axis = np.array([0.0, 1.0, 0.0])
        anchor_quat = rotmat_to_quat(anchor).squeeze()
        root_orient_target_quat = rotmat_to_quat(root_orient_target).squeeze()
        _, theta = quat_closest(root_orient_target_quat, anchor_quat, up_axis)
        theta = np.expand_dims(theta, axis=-1)
        up_axis_repeat = np.repeat(up_axis[None, :], theta.shape[0], axis=0)
        dR_aaxis = theta * up_axis_repeat
        dR = aaxis_to_rotmat(dR_aaxis) # [1, 3, 3]
        dR = np.expand_dims(dR, axis=-3)
        root_orient_dest = dR @ anchor
        
        # rotated precomputed root_trans_disp
        root_trans_disp = self.decoded_accum_root_trans_disp[next_idx:next_idx+1]
        root_trans_disp = np.swapaxes(root_trans_disp, -1, -2)
        root_trans_disp_anchor = anchor_inv @ root_trans_disp # need to convert reference frame first
        root_trans_disp_rotated = root_orient_dest @ root_trans_disp_anchor
        root_trans_disp_rotated = np.swapaxes(root_trans_disp_rotated, -1, -2)
        new_root_pos = accum_root_pos_exp + root_trans_disp_rotated
        
        forward = new_root_pos.squeeze() - accum_root_pos
        forward[1] = 0 # on the ground
        forward = forward / np.linalg.norm(forward) # normalize
        
        # align root orient
        root_orient_anchor = anchor_inv @ root_orient
        root_orient_rotated = root_orient_dest @ root_orient_anchor

        new_root_trf = np.eye(4)
        new_root_trf[..., :3, 3] = new_root_pos.squeeze()
        new_root_trf[..., :3, :3] = root_orient_rotated[0, -1]
        
        control_cost = utils.cos_dist(forward, control)
        cost_total = cost + self.weights["control"] * control_cost
        return cost_total, new_root_trf
    
    def precompute_all_pair_cost(
        self,
        weights: dict = None
    ):
        if weights is None:
            weights = self.weights
        
        print("Precomputing all costs...")
        N = self.adj.shape[0]
        all_pair_cost = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                c = self.compute_cost(i, j, weights)
                all_pair_cost[i][j] = c                
        self.all_pair_cost = all_pair_cost.astype(np.float32)
    
    def save_cost(
        self,
        save_dir: str
    ):
        save_path = f"{save_dir}/all_pair_cost.npy"
        np.save(save_path, self.all_pair_cost)
    
    def load_cost(
        self,
        load_dir: str
    ):
        load_path = f"{load_dir}/all_pair_cost.npy"
        cost = np.load(load_path)
        self.all_pair_cost = cost
        
    def sample_start_goal(self):
        nodes = self.largest_comp_nodes
        samples = np.random.choice(nodes, size=2, replace=True)
        return samples[0], samples[1] # start, goal node

    def astar(
        self,
        start,
        goal
    ):
        """
        A* search algorithm for directed graphs using an adjacency list.

        Assumes:
            self.graph[node] = [(neighbor, edge_cost), ...]

        Args:
            start (int): Index of the start node.
            goal (int): Index of the goal node.

        Returns:
            path (list): Shortest path from start to goal.
            cost (float): Cost of the shortest path.
            visited (set): Visited nodes.
        """
        graph_adj_list = self.graph_adj_list
        N = len(graph_adj_list)
        inf = float("inf")
        compute_cost = self.compute_cost

        open_set = []
        heapq.heappush(open_set, (compute_cost(start, goal, self.weights), 0, start, (start,)))

        visited = set()
        best_g = [inf] * N
        best_g[start] = 0

        best_path = [None] * N
        best_path[start] = (start,)

        while open_set:
            _, g, node, path = heapq.heappop(open_set)

            if node in visited:
                continue
            visited.add(node)

            if node == goal:
                return list(path), g, visited

            for neighbor, edge in graph_adj_list[node]:
                new_g = g + edge
                new_path = path + (neighbor,)

                if new_g < best_g[neighbor]:
                    best_g[neighbor] = new_g
                    best_path[neighbor] = new_path
                    f = new_g + compute_cost(neighbor, goal, self.weights)
                    heapq.heappush(open_set, (f, new_g, neighbor, new_path))
        return None, inf, visited

    def astar_control(
        self, 
        start, 
        goal,
        control,
        init_root_trf # last root trf of targetA
    ):
        N = self.graph.shape[0]
        open_set = []
        init_cost, _ = self.compute_cost_control(start, goal, init_root_trf, control)
        heapq.heappush(open_set, (0 + init_cost, 0, start, [start], init_root_trf))
        
        
        visited = set()
        best_g = {i: float("inf") for i in range(N)}
        best_g[start] = 0
        
        while open_set:
            _, g, node, path, root_trf = heapq.heappop(open_set) # get node with lowest f-score
            
            
            if node in visited:
                continue
            visited.add(node)
            
            if node == goal:
                return path, g, visited # return shortest path, cost and visited nodes
            
            for neighbor in range(N):
                if self.graph[node, neighbor] == float("inf"): # no edge
                    continue
                
                # compute g-score
                edge, neighbor_root_trf = self.compute_cost_control(node, neighbor, root_trf, control)
                new_g = g + edge
                
                if new_g < best_g[neighbor]: # if found better path, record heuristic
                    best_g[neighbor] = new_g
                    heu, new_root_trf = self.compute_cost_control(neighbor, goal, neighbor_root_trf, control, edge=False)
                    f = new_g + heu # compute f-score
                    heapq.heappush(open_set, (f, new_g, neighbor, path + [neighbor], new_root_trf))
                    
        return None, float("inf"), visited # no path found