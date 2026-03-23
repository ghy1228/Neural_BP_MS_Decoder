import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset

class CodeParam:
    """Holds LDPC code parameters and protograph info."""
    def __init__(
        self, H, N, M, E,
        CN_deg, VN_deg, cn_max_deg,
        cn_to_edge, vn_to_edge,
        edge_to_VN, edge_to_CN, edge_to_ext_edge,
        M_proto, N_proto, E_proto,
        z_value, proto_matrix,
        puncturing_idx=None, shortening_idx=None
    ):
        self.h_matrix = H
        self.N = N
        self.M = M
        self.E = E
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
        self.puncturing_idx = puncturing_idx if puncturing_idx is not None else []
        self.shortening_idx = shortening_idx if shortening_idx is not None else []

        # Effective parameters considering puncturing/shortening
        num_punctured = len(self.puncturing_idx)
        num_shortened = len(self.shortening_idx)
        self.N_effective = N - num_punctured - num_shortened  # transmitted bits
        self.K_effective = N - M - num_shortened              # information bits
        self.code_rate = self.K_effective / self.N_effective if self.N_effective > 0 else (N - M) / N

class CustomDataset(Dataset):
    """Simple dataset containing only LLR samples."""
    def __init__(self, llrs):
        super().__init__()
        self.llrs = llrs

    def __getitem__(self, idx):
        return torch.tensor(self.llrs[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.llrs)

def read_alist_file(filename):
    """
    Read LDPC parity check matrix from alist format file.
    Returns H matrix and basic parameters.
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # First line: N M (number of variable nodes, check nodes)
    N, M = map(int, lines[0].split())
    
    # Second line: max_VN_degree max_CN_degree
    max_vn_deg, max_cn_deg = map(int, lines[1].split())
    
    # Third line: VN degrees
    vn_degrees = list(map(int, lines[2].split()))
    
    # Fourth line: CN degrees  
    cn_degrees = list(map(int, lines[3].split()))
    
    # Initialize H matrix
    H = np.zeros((M, N), dtype=int)
    
    # Read VN connections (lines 4 to 4+N-1)
    for i in range(N):
        connections = list(map(int, lines[4 + i].split()))
        # Remove zeros and convert to 0-based indexing
        connections = [c - 1 for c in connections if c > 0]
        for cn_idx in connections:
            if 0 <= cn_idx < M:
                H[cn_idx, i] = 1
    
    return H

def init_parameter(filename, z_factor, puncturing_idx=None, shortening_idx=None):
    """
    Load matrix and build full H from either proto (.txt) or alist (.alist) format.
    Return a CodeParam object with edges, degrees, etc.
    """
    # Try alist format first
    alist_path = f"./BaseGraph/{filename}.alist"
    txt_path = f"./BaseGraph/{filename}.txt"

    if os.path.exists(alist_path):
        # Alist format - assume z_factor = 1
        z_factor = 1
        H = read_alist_file(alist_path)
        M, N = H.shape

        # For alist, we don't have a proto matrix, so create a dummy one
        M_proto, N_proto = M, N
        E_proto = np.sum(H)
        proto_mat = H.copy()  # Use H itself as proto matrix

    elif os.path.exists(txt_path):
        # Proto format (.txt)
        proto_mat = np.loadtxt(txt_path, dtype=int, delimiter='\t')
        M_proto, N_proto = proto_mat.shape
        E_proto = np.count_nonzero(proto_mat != -1)

        # Build full H matrix from proto matrix
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

    else:
        raise FileNotFoundError(f"Neither {alist_path} nor {txt_path} found")

    # Common processing for both formats
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

    # Build edge_to_ext_edge matrix
    edge_to_ext_edge = -1 * np.ones((E, cn_max_deg-1), dtype=np.int32)

    for c in range(M):
        edges_c = cn_to_edge[c]  # edges connected to this CN
        for e in edges_c:
            # list of other edges excluding e itself
            others = [ex for ex in edges_c if ex != e]
            # fill only the front part of d_max-1, leave the rest as -1
            for i, e2 in enumerate(others):
                edge_to_ext_edge[e, i] = e2

    return CodeParam(
        H=H,
        N=N, M=M, E=E,
        CN_deg=CN_deg, VN_deg=VN_deg, cn_max_deg=cn_max_deg,
        cn_to_edge=cn_to_edge, vn_to_edge=vn_to_edge,
        edge_to_VN=edge_to_VN, edge_to_CN=edge_to_CN,
        edge_to_ext_edge=edge_to_ext_edge,
        M_proto=M_proto, N_proto=N_proto,
        E_proto=E_proto, z_value=z_factor,
        proto_matrix=proto_mat,
        puncturing_idx=puncturing_idx, shortening_idx=shortening_idx
    )

def create_random_samples(sample_num, code_rate, SNR_array, rng, N,
                          puncturing_idx=None, shortening_idx=None, clip_llr=20.0):
    """
    Generate LLR samples for AWGN, code_rate for noise variance calc.

    Args:
        sample_num: Number of samples to generate
        code_rate: Code rate (used for noise variance calculation)
        SNR_array: List of SNR values (in dB)
        rng: Random number generator
        N: Codeword length
        puncturing_idx: List of punctured bit indices (LLR = 0)
        shortening_idx: List of shortened bit indices (LLR = clip_llr)
        clip_llr: LLR value assigned to shortened bits

    Returns:
        llrs: numpy array of shape (sample_num, N)
    """
    if sample_num <= 0:
        return np.array([])

    if puncturing_idx is None:
        puncturing_idx = []
    if shortening_idx is None:
        shortening_idx = []

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

        # Apply puncturing: LLR = 0 (no channel information)
        for idx in puncturing_idx:
            if 0 <= idx < N:
                llr[idx] = 0.0

        # Apply shortening: LLR = clip_llr (known to be 0, so large positive LLR)
        for idx in shortening_idx:
            if 0 <= idx < N:
                llr[idx] = clip_llr

        llrs.append(llr)
    return np.array(llrs, dtype=float)
