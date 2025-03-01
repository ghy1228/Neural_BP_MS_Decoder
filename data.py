import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset

class CodeParam:
    """Holds LDPC code parameters and protograph info."""
    def __init__(
        self, H,N, M, E, code_rate, 
        CN_deg, VN_deg, cn_max_deg, 
        cn_to_edge, vn_to_edge,
        edge_to_VN, edge_to_CN, edge_to_ext_edge,
        M_proto, N_proto, E_proto, 
        z_value, proto_matrix
    ):
        self.h_matrix = H
        self.N = N
        self.M = M
        self.E = E
        self.code_rate = code_rate
        self.CN_deg = CN_deg
        self.VN_deg = VN_deg
        self.cn_max_deg = cn_max_deg
        self.cn_to_edge = cn_to_edge
        self.vn_to_edge = vn_to_edge
        self.edge_to_VN = edge_to_VN
        self.edge_to_CN = edge_to_CN
        self.edge_to_ext_edge = edge_to_ext_edge
        self.M_proto = M_proto
        self.N_proto = N_proto
        self.E_proto = E_proto
        self.z_value = z_value
        self.proto_matrix = proto_matrix

class CustomDataset(Dataset):
    """Simple dataset containing only LLR samples."""
    def __init__(self, llrs):
        super().__init__()
        self.llrs = llrs

    def __getitem__(self, idx):
        return torch.tensor(self.llrs[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.llrs)

def init_parameter(filename, z_factor, CN_mode):
    """
    Load proto-matrix and build full H. 
    Return a CodeParam object with edges, degrees, etc.
    """
    path = f"./BaseGraph/{filename}.txt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    proto_mat = np.loadtxt(path, dtype=int, delimiter='\t')
    M_proto, N_proto = proto_mat.shape
    E_proto = np.count_nonzero(proto_mat != -1)

    blocks = []
    for r in range(M_proto):
        row_blocks = []
        for c in range(N_proto):
            val = proto_mat[r,c]
            if val == -1:
                blk = np.zeros((z_factor,z_factor), dtype=int)
            else:
                blk = np.zeros((z_factor,z_factor), dtype=int)
                for rr in range(z_factor):
                    cc = (rr + val) % z_factor
                    blk[rr, cc] = 1
            row_blocks.append(blk)
        row_cat = np.hstack(row_blocks)
        blocks.append(row_cat)

    H = np.vstack(blocks)
    M, N = H.shape
    code_rate = 1 - (M / N)
    CN_deg = np.sum(H, axis=1)
    VN_deg = np.sum(H, axis=0)
    E = int(np.sum(VN_deg))
    cn_to_edge = [[] for _ in range(M)]
    vn_to_edge = [[] for _ in range(N)]
    edge_to_VN = np.zeros(E, dtype=int)
    edge_to_CN = np.zeros(E, dtype=int)
    cn_max_deg = max(CN_deg)

    idx = 0
    for m_ in range(M):
        for n_ in range(N):
            if H[m_,n_] == 1:
                cn_to_edge[m_].append(idx)
                vn_to_edge[n_].append(idx)
                edge_to_VN[idx] = n_
                edge_to_CN[idx] = m_
                idx += 1

    # 미리 (E, d_max-1) 짜리 -1로 초기화
    edge_to_ext_edge = -1 * np.ones((E, cn_max_deg-1), dtype=np.int32)

    # 채우기
    for c in range(M):
        edges_c = cn_to_edge[c]  # 이 CN에 연결된 edge들
        for e in edges_c:
            # e 자신을 제외한 edge 목록
            others = [ex for ex in edges_c if ex != e]
            # d_max-1 중 앞쪽만 실제 값, 남으면 -1로 둔다
            for i, e2 in enumerate(others):
                edge_to_ext_edge[e, i] = e2

    
    return CodeParam(
        H=H,
        N=N, M=M, E=E, code_rate=code_rate,
        CN_deg=CN_deg, VN_deg=VN_deg, cn_max_deg = cn_max_deg,
        cn_to_edge=cn_to_edge, vn_to_edge=vn_to_edge,
        edge_to_VN=edge_to_VN, edge_to_CN=edge_to_CN,
        edge_to_ext_edge = edge_to_ext_edge,
        M_proto=M_proto, N_proto=N_proto,
        E_proto=E_proto, z_value=z_factor,
        proto_matrix=proto_mat
    )

def create_random_samples(sample_num, code_rate, SNR_array, rng, N):
    """
    Generate LLR samples for AWGN, code_rate for noise variance calc.
    """
    if sample_num <= 0:
        return np.array([])
    llrs = []
    n_snr = len(SNR_array)
    for i in range(sample_num):
        snr_db = SNR_array[i % n_snr]
        snr_lin = 10 ** (snr_db / 10.0)
        noise_var = 1.0 / (2.0 * snr_lin * code_rate)
        noise_std = math.sqrt(noise_var)
        tx = np.ones(N, dtype=float)
        noise = rng.normal(0, noise_std, N)
        rx = tx + noise
        llr = (2.0 / noise_var) * rx
        llrs.append(llr)
    return np.array(llrs, dtype=float)
