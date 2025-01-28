# model.py
import torch
import torch.nn as nn
import math
import pdb
import os, logging
import numpy as np


class LDPCNetwork(nn.Module):
    """
    Simple LDPC decoder example.
      - sharing:     [cn_weight_sharing, ucn_weight_sharing, ch_weight_sharing]
      - cn_mode:     'sequential' (for-loop per CN) or 'parallel' (use edge_to_ext_edge).
    """

    def __init__(self, code_param, args, device=None):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.N = code_param.N
        self.M = code_param.M
        self.E = code_param.E
        self.z_factor = code_param.z_value
        
        # Basic index arrays (on GPU)
        self.h_matrix   = torch.from_numpy(code_param.h_matrix).float().to(self.device)
        self.edge_to_vn = torch.from_numpy(code_param.edge_to_VN).long().to(self.device)
        self.edge_to_cn = torch.from_numpy(code_param.edge_to_CN).long().to(self.device)
        self.cn_to_edge = code_param.cn_to_edge
        self.vn_to_edge = code_param.vn_to_edge

        # cn_mode controls how the CN update is done: 'parallel' or 'sequential'
        self.cn_mode = args.cn_mode
        

        # If parallel, we have an index list edge_to_ext_edge (E x (d_max-1))
        if self.cn_mode == 'parallel':
            self.edge_to_ext_edge = torch.from_numpy(code_param.edge_to_ext_edge).int().to(self.device)
        else:
            self.edge_to_ext_edge = None

        self.N_proto = code_param.N_proto
        self.M_proto = code_param.M_proto
        self.E_proto = code_param.E_proto

        self.q_bit = args.q_bit
        self.init_cn_weight = args.init_cn_weight
        self.init_ch_weight = args.init_ch_weight
        self.init_cn_bias  = args.init_cn_bias
        self.iters_max   = args.iters_max
        self.clip_llr    = args.clip_llr
        self.loss_function = args.loss_function
        self.loss_option   = args.loss_option
        self.decoding_type = args.decoding_type
        self.batch_size  = args.batch_size
        self.sharing     = args.sharing
        self.systematic  = args.systematic

        # Initialize optional learnable weights
        self.cn_weight  = self._init_weight(self.sharing[0], prefix="cn")
        self.ucn_weight = self._init_weight(self.sharing[1], prefix="ucn")
        self.ch_weight  = self._init_weight(self.sharing[2], prefix="ch")
        self.cn_bias  = self._init_weight(self.sharing[3], prefix="cn_b")
        self.ucn_bias = self._init_weight(self.sharing[4], prefix="ucn_b")

    def _init_weight(self, s_val, prefix):
        """
        Create learnable weight parameters based on sharing scheme.
        s_val indicates the type of sharing (0=none, 1=per-edge-proto, etc.).
        """
        if s_val == 0:
            return None
        if prefix in ["cn", "ucn"]:
            if   s_val == 1:
                w = torch.ones(self.iters_max, self.E_proto, device=self.device) * self.init_cn_weight
            elif s_val == 2:
                w = torch.ones(self.iters_max, self.M_proto, device=self.device) * self.init_cn_weight
            elif s_val == 3:
                w = torch.ones(self.iters_max, device=self.device) * self.init_cn_weight
            elif s_val == 4:
                w = torch.ones(self.E_proto, device=self.device) * self.init_cn_weight
            elif s_val == 5:
                w = torch.ones(self.M_proto, device=self.device) * self.init_cn_weight
            else:
                return None
        elif prefix in ["cn_b", "ucn_b"]:
            if   s_val == 1:
                w = torch.ones(self.iters_max, self.E_proto, device=self.device) * self.init_cn_bias
            elif s_val == 2:
                w = torch.ones(self.iters_max, self.M_proto, device=self.device) * self.init_cn_bias
            elif s_val == 3:
                w = torch.ones(self.iters_max, device=self.device) * self.init_cn_bias
            elif s_val == 4:
                w = torch.ones(self.E_proto, device=self.device) * self.init_cn_bias
            elif s_val == 5:
                w = torch.ones(self.M_proto, device=self.device) * self.init_cn_bias
            else:
                return None
        elif prefix == "ch":
            if   s_val == 2:
                w = torch.ones(self.iters_max, self.N_proto, device=self.device) * self.init_ch_weight
            elif s_val == 3:
                w = torch.ones(self.iters_max, device=self.device) * self.init_ch_weight
            elif s_val == 5:
                w = torch.ones(self.N_proto, device=self.device) * self.init_cn_weight
            else:
                return None
        else:
            return None
        return nn.Parameter(w)

    def forward(self, llr_in, return_dec_llr=False):
        """
        Main decoding loop for iters_max iterations.
        Returns the final decoded LLR if requested.
        """
        v2c_llr = torch.zeros(self.batch_size, self.E, device=self.device)
        c2v_llr = torch.zeros(self.batch_size, self.E, device=self.device)
        sum_llr = torch.zeros(self.batch_size, self.N, device=self.device)
        dec_llr = llr_in

        dec_llr_list = []
        for it in range(self.iters_max):
            syndrome = self._get_syndrome(dec_llr)
            # Optional channel-weight application
            w_ch = self._apply_ch_weight(llr_in, it)

            # VN update
            v2c_llr = self._vn_update(w_ch, c2v_llr, sum_llr)
            v2c_llr = self._quantize_llr(v2c_llr, self.q_bit)
            

            # CN update (SP or MS), either parallel or sequential
            if self.decoding_type == 'SP':  # SP
                if self.cn_mode == 'parallel':
                    c2v_unweighted = self._cn_update_SP_par(v2c_llr)
                else:
                    c2v_unweighted = self._cn_update_SP_seq(v2c_llr)
            else:  # MS
                if self.cn_mode == 'parallel':
                    c2v_unweighted = self._cn_update_MS_par(v2c_llr)
                else:
                    c2v_unweighted = self._cn_update_MS_seq(v2c_llr)

            # Optional CN weighting and sum of CN->VN messages
            c2v_llr_w = self._apply_cn_weight(c2v_unweighted, it, syndrome)
            c2v_llr_wb = self._apply_cn_bias(c2v_llr_w, it, syndrome)
            c2v_llr = self._quantize_llr(c2v_llr_wb, self.q_bit)
            sum_llr = self._compute_sum_llr(c2v_llr, sum_llr)

            # Hard-decision LLR for output
            dec_llr = self._decision(llr_in, c2v_llr)
            dec_llr_list.append(dec_llr)

        loss_val = self._compute_loss_all(dec_llr_list)
        if return_dec_llr:
            return loss_val, dec_llr_list[-1]
        return loss_val
    
    def _quantize_llr(self, llr, q_bit, step_size=0.5):
        """
        Quantize LLR values based on q_bit.
        If q_bit == 0, perform clipping using self.clip_llr.
        If q_bit >= 1, perform quantization with step size.
        """
        if q_bit == 0:
            quantized = torch.clamp(llr, -self.clip_llr, self.clip_llr)
        else:
            max_val = (2 ** (q_bit - 1) - 1) * step_size
            min_val = -max_val
            quantized = torch.round(llr / step_size) * step_size  # Apply quantization step
            quantized = torch.clamp(llr, min_val, max_val) 
        
        return quantized.detach() + (llr - llr.detach())

    def _apply_ch_weight(self, llr_in, it_idx):
        """
        Apply channel (ch) weights if needed (e.g. per-iteration, scalar or proto-based).
        """
        if self.ch_weight is None:
            return llr_in
        s_val = self.sharing[2]
        if s_val == 2 or s_val == 5:
            v = torch.arange(self.N, device=self.device)
            proto_col = v // (self.z_factor)
            if s_val == 2:
                w_iter = self.ch_weight[it_idx, proto_col]
                w_iter = w_iter.unsqueeze(0).expand(llr_in.size(0), -1)
            else:
                w_iter = self.ch_weight[proto_col]
                w_iter = w_iter.unsqueeze(0).expand(llr_in.size(0), -1)
            return llr_in * w_iter
        elif s_val == 3:
            scalar = self.ch_weight[it_idx]
            return llr_in * scalar
        
        return llr_in

    def _apply_cn_weight(self, c2v_llr, it_idx, syndrome):
        """
        Apply CN weights if sharing is enabled.
        Clamps final LLR within [-clip_llr, clip_llr].
        """

        def compute_weight(sharing_type, weight, iter, edge_proto, edge_to_cn):
            if sharing_type == 1:
                return weight[iter][edge_proto]
            elif sharing_type == 2:
                return weight[iter][edge_to_cn]
            elif sharing_type == 3:
                return weight[iter]
            elif sharing_type == 4:
                return weight[edge_proto]
            elif sharing_type == 5:
                return weight[edge_to_cn]
            else:
                return torch.ones_like(c2v_llr)

        
        c2v_new = c2v_llr.clone()
        edge_to_cn_proto = self.edge_to_cn // self.z_factor
        edge_proto = torch.arange(self.E, device=self.device) // (self.z_factor) #Proto Sharing
        if self.cn_weight is not None and self.ucn_weight is not None:
            syn_e = syndrome[:, self.edge_to_cn]
            
            # Compute weights for synd=0 and synd=1
            w_edge_0 = compute_weight(self.sharing[0], self.cn_weight, it_idx, edge_proto, edge_to_cn_proto)
            w_edge_1 = compute_weight(self.sharing[1], self.ucn_weight, it_idx, edge_proto, edge_to_cn_proto)
            
        
            # Combine weights based on syndromes
            mask_1 = (syn_e == 1).float()
            mask_0 = 1.0 - mask_1
            w_edge = w_edge_0 * mask_0 + w_edge_1 * mask_1
            c2v_new *= w_edge

        elif self.cn_weight is not None:
            w_edge = compute_weight(self.sharing[0], self.cn_weight, it_idx, edge_proto, edge_to_cn_proto)
            c2v_new *= w_edge

        return c2v_new
    
    def _apply_cn_bias(self, c2v_llr_w, it_idx, syndrome):
        """
        Apply CN bias with given bias parameters and handle different sharing types.
        Bias is applied as a subtraction, and negative LLRs after bias are set to 0.
        Handles cases where biases may or may not be provided and adjusts for sharing type.
        """
        def compute_bias(sharing_type, bias, edge_proto, edge_to_cn):
            """
            Compute bias based on the sharing type.
            """
            if sharing_type == 1:
                return bias[it_idx][edge_proto]  # Iteration and proto-specific bias
            elif sharing_type == 2:
                return bias[it_idx][edge_to_cn]  # Iteration and CN-specific bias
            elif sharing_type == 3:
                return bias[it_idx]  # Iteration-specific bias
            elif sharing_type == 4:
                return bias[edge_proto]  # Proto-specific bias
            elif sharing_type == 5:
                return bias[edge_to_cn]  # CN-specific bias
            else:
                return torch.zeros_like(c2v_llr_w)  # Default to zero bias if sharing type is invalid

        c2v_new = c2v_llr_w.clone()  # Clone the input tensor to avoid modifying it directly
        edge_to_cn_proto = self.edge_to_cn // self.z_factor
        edge_proto = torch.arange(self.E, device=self.device) // self.z_factor  # Proto Sharing

        # Check if biases are provided
        if self.cn_bias is not None and self.ucn_bias is not None:
            # Compute biases for synd=0 and synd=1
            syn_e = syndrome[:, self.edge_to_cn]
            bias_0 = compute_bias(self.sharing[3], self.cn_bias, edge_proto, edge_to_cn_proto)  # Bias for synd=0
            bias_1 = compute_bias(self.sharing[4], self.ucn_bias, edge_proto, edge_to_cn_proto)  # Bias for synd=1

            # Combine biases based on syndromes
            mask_1 = (syn_e == 1).float()
            mask_0 = 1.0 - mask_1
            bias = bias_0 * mask_0 + bias_1 * mask_1

        elif self.cn_bias is not None:
            # Only cn_bias is available
            bias = compute_bias(self.sharing[3], self.cn_bias, edge_proto, edge_to_cn_proto)

        elif self.ucn_bias is not None:
            # Only ucn_bias is available
            syn_e = syndrome[:, self.edge_to_cn]
            bias = compute_bias(self.sharing[4], self.ucn_bias, edge_proto, edge_to_cn_proto) * (syn_e == 1).float()

        else:
            # No bias is available, return the input as is
            return c2v_new

        # Apply biases
        c2v_abs = torch.abs(c2v_new)  # Compute the absolute value of LLRs
        c2v_abs = c2v_abs - bias  # Subtract bias
        c2v_abs = torch.clamp(c2v_abs, min=0)  # Set negative values to 0

        # Restore original sign
        c2v_new = c2v_abs * torch.sign(c2v_new)

        return c2v_new



    def _vn_update(self, w_ch, c2v_llr, sum_llr):
        """
        VN update: v2c_llr = (channel LLR + sum of other c2v) - current c2v.
        Clamps at clip_llr.
        """
        v_idx = self.edge_to_vn
        wch_g = w_ch.gather(1, v_idx.unsqueeze(0).expand(w_ch.size(0), -1))
        sum_g = sum_llr.gather(1, v_idx.unsqueeze(0).expand(sum_llr.size(0), -1))
        out = wch_g + sum_g - c2v_llr
        return out

    def _cn_update_SP_seq(self, v2c_llr):
        """
        Sequential SP: loop over each CN, compute product of tanh(0.5*x).
        Then extrinsic = total / self, and convert via 2 * atanh.
        """
        c2v_new = torch.zeros_like(v2c_llr)
        eps = 1e-12
        for c in range(self.M):
            edges_c = self.cn_to_edge[c]
            if len(edges_c) == 0:
                continue
            x = v2c_llr[:, edges_c] * 0.5
            x = torch.tanh(x)
            x = torch.where(x == 0, x.new_tensor(eps), x)
            p_all = torch.prod(x, dim=1)
            out = p_all.unsqueeze(1) / x
            out = torch.clamp(out, -0.999999, 0.999999)
            out = 2.0 * 0.5 * torch.log((1 + out)/(1 - out + eps))
            c2v_new[:, edges_c] = out
        return c2v_new

    def _cn_update_SP_par(self, v2c_llr):
        """
        Parallel SP using [E, d_max-1] indices:
          1) x_tanh = tanh(0.5 * v2c_llr)
          2) gather extrinsic edges -> product
          3) convert to c2v via 2 * atanh(product).
        """
        eps = 1e-12
        c2v_new = torch.zeros_like(v2c_llr)

        # x_tanh
        x = 0.5 * v2c_llr
        x_tanh = torch.tanh(x)
        x_tanh = torch.where(x_tanh == 0, x_tanh.new_tensor(eps), x_tanh)

        ext_idx = self.edge_to_ext_edge  # [E, d_max-1], or -1 for padding
        x_tanh_3d = x_tanh[:, ext_idx]   # => [B, E, d_max-1]

        mask_invalid = (ext_idx < 0)
        x_tanh_3d = torch.where(
            mask_invalid.unsqueeze(0),
            x_tanh_3d.new_tensor(1.0),  # 1.0 for product neutral
            x_tanh_3d
        )

        prod_ext = torch.prod(x_tanh_3d, dim=2)  # => [B, E]

        out = torch.clamp(prod_ext, -0.999999, 0.999999)
        out = 0.5 * torch.log((1.0 + out)/(1.0 - out + eps)) * 2.0
        c2v_new = out
        return c2v_new

    def _cn_update_MS_seq(self, v2c_llr):
        """
        Sequential MS: loop over each CN, find min1, min2, etc. (top-2 min approach).
        """
        bsz = v2c_llr.size(0)
        c2v_new = torch.zeros_like(v2c_llr)

        for c in range(self.M):
            edges_c = self.cn_to_edge[c]
            if len(edges_c) == 0:
                continue
            vals = v2c_llr[:, edges_c]
            absvals = torch.abs(vals)
            sgnvals = torch.sign(vals)
            tot_sign = torch.prod(
                torch.where(sgnvals == 0, torch.ones_like(sgnvals), sgnvals),
                dim=1
            )
            # top-2 min
            top2, idx2 = torch.topk(absvals, 2, dim=1, largest=False)
            mag1 = top2[:, 0]
            mag2 = top2[:, 1]
            pos = idx2[:, 0]  # argmin1
            row_idx = torch.arange(vals.size(1), device=vals.device).unsqueeze(0).expand(bsz, -1)
            mg = mag1.unsqueeze(1).expand(-1, vals.size(1)).clone()
            mg[row_idx == pos.unsqueeze(1)] = mag2
            s_j = torch.where(vals < 0, -tot_sign.unsqueeze(1), tot_sign.unsqueeze(1))
            c2v_new[:, edges_c] = s_j * mg

        return c2v_new

    def _cn_update_MS_par(self, v2c_llr):
        """
        Parallel MS using [E, d_max-1]:
          1) gather abs and sign from extrinsic edges
          2) min across dim=2, product of sign across dim=2
          3) combine to form c2v_new
        """
        c2v_new = torch.zeros_like(v2c_llr)

        ext_idx = self.edge_to_ext_edge  # [E, d_max-1]
        absvals = v2c_llr.abs()
        sgnvals = torch.sign(v2c_llr)

        # Gather extrinsic abs -> [B, E, d_max-1]
        extrinsic_abs = absvals[:, ext_idx]
        big_val = absvals.new_tensor(1e9)
        mask_invalid = (ext_idx < 0)
        extrinsic_abs = torch.where(
            mask_invalid.unsqueeze(0),
            big_val,
            extrinsic_abs
        )
        min_abs = extrinsic_abs.min(dim=2).values  # => [B, E]

        # Gather extrinsic sign -> [B, E, d_max-1]
        extrinsic_sgnvals = sgnvals[:, ext_idx]
        extrinsic_sgnvals = torch.where(
            mask_invalid.unsqueeze(0),
            extrinsic_sgnvals.new_tensor(1.0),
            extrinsic_sgnvals
        )
        sign_prod = torch.prod(extrinsic_sgnvals, dim=2)  # => [B, E]

        out = min_abs * sign_prod

        c2v_new = out
        return c2v_new

    def _compute_sum_llr(self, c2v_llr, sum_llr):
        """
        Accumulate CN->VN messages to form sum_llr (used in VN update).
        """
        nxt = torch.zeros_like(sum_llr)
        nxt.index_add_(1, self.edge_to_vn, c2v_llr)
        return nxt
    
    def _get_syndrome(self,dec_llr):
        if self.sharing[1] is not None:
            dec_bit = (dec_llr < 0).float()
            syndrome = torch.fmod(torch.matmul(dec_bit,self.h_matrix.T),2)
            return syndrome
        else:
            return None
        

    def _decision(self, llr_in, c2v_llr):
        """
        Final LLR combination: channel LLR + sum of c2v.
        Clamps the output.
        """
        dec_llr = llr_in.clone()
        sum_c2v = torch.zeros_like(dec_llr)
        sum_c2v.index_add_(1, self.edge_to_vn, c2v_llr)
        dec_llr += sum_c2v
        dec_llr.clamp_(-self.clip_llr, self.clip_llr)
        return dec_llr

    def _compute_loss_all(self, dec_llr_list):
        """
        Aggregate loss across all iterations or just the last.
        """
        losses = []
        for dec_llr in dec_llr_list:
            losses.append(self._compute_loss_one_iter(dec_llr))
        if self.loss_option == 'multi':
            return torch.stack(losses, dim=0).mean()
        else:
            return losses[-1]

    def _compute_loss_one_iter(self, dec_llr):
        """
        Compute loss for one iteration, considering only the systematic bits
        if self.systematic is 'on'.
        """
        if hasattr(self, 'systematic') and self.systematic == 'on':
            # Select only the first self.N - self.M values for each batch
            dec_llr = dec_llr[:, :self.N - self.M]

        if self.loss_function == 'BCE':
            # Binary Cross-Entropy Loss
            return -torch.log(1.0 - torch.sigmoid(-dec_llr) + 1e-12).mean()
        elif self.loss_function == 'Soft_BER':
            # Soft Bit Error Rate
            return torch.sigmoid(-dec_llr).mean()
        elif self.loss_function == 'FER':
            # Frame Error Rate with Straight-Through Estimator (STE)
            min_val = torch.min(dec_llr, dim=1).values
            sign_val = torch.sign(min_val)
            inv_exp_val = 2.0 / (1.0 + torch.exp(-min_val)) - 1.0
            # Forward uses sign, backward uses inv_exp
            ste_val = inv_exp_val + (sign_val - inv_exp_val).detach()

            fer = 0.5 * (1.0 - ste_val)
            return fer.mean()
        else:
            # Default to zero loss if no valid loss function is specified
            return torch.zeros(1, device=dec_llr.device)


    def clamp_weights(self):
        """
        weights clamp.
        """
        # 1) cn_weight, ch_weight, cnd_weight, chd_weight: [0, 3]
        if self.cn_weight is not None:
            self.cn_weight.data.clamp_(0.0, 3.0)
        if self.ucn_weight is not None:
            self.ucn_weight.data.clamp_(0.0, 3.0)
        if self.ch_weight is not None:
            self.ch_weight.data.clamp_(0.0, 3.0)
        if self.cn_bias is not None:
            self.cn_bias.data.clamp_(-self.clip_llr, self.clip_llr)
        if self.ucn_bias is not None:
            self.ucn_bias.data.clamp_(-self.clip_llr, self.clip_llr)



    def load_init_weights_from_file(net, args):
    
        path = f"./Weights/{args.filename}_In_Weight_Iter{args.iters_max}.txt"
        if not os.path.exists(path):
            logging.warning(f"No init weight file: {path}. Using default init=1.")
            return

        logging.info(f"Loading initial weights from {path}")
        with open(path, "r") as f:
            lines = [line.strip() for line in f]

        weight_dict = {
            "cn_weight":   None,
            "ucn_weight":   None,
            "ch_weight":   None,
            "cn_bias":   None,
            "ucn_bias":   None,
        }

        idx = 0
        current_key = None
        buffer_list = []
        
        while idx < len(lines):
            line = lines[idx]
            idx += 1

            if line.endswith(":"):
                current_key = line.replace(":", "")
                buffer_list = []
            elif line == "":
                if current_key and buffer_list:
                    arr = np.array(buffer_list, dtype=np.float32)
                    weight_dict[current_key] = arr
                buffer_list = []
            elif line != "None":
                floats = list(map(float, line.split('\t')))
                buffer_list.append(floats)

        if buffer_list and current_key:
            arr = np.array(buffer_list, dtype=np.float32)
            weight_dict[current_key] = arr

        for name_str, arr in weight_dict.items():
            param = getattr(net, name_str, None)
            if arr is None:
                setattr(net, name_str, None)
            else:
                if param is not None:
                    p_data = param.data
                    
                    if arr.size == p_data.numel():
                        arr = arr.reshape(p_data.shape)

                    if p_data.shape == arr.shape:
                        p_data.copy_(torch.from_numpy(arr))
                    else:
                        logging.warning(f"Shape mismatch for {name_str}. "
                                        f"Expected {p_data.shape}, got {arr.shape}.")

    def freeze_first_iters(net, args):
        f_iter = args.fixed_iter
        if f_iter <= 0:
            return

        for w_name in ["cn_weight", "ucn_weight", "ch_weight","cn_bias","ucn_bias"]:
            w_param = getattr(net, w_name, None)
            if w_param is None:
                continue

            if w_param.ndim == 2:
                w_param.requires_grad = True
                def hook_fn(grad, fi=f_iter):
                    grad[:fi, :] = 0
                    return grad
                w_param.register_hook(hook_fn)
            elif w_param.ndim == 1:
                w_param.requires_grad = True
                def hook_fn(grad, fi=f_iter):
                    grad[:fi] = 0
                    return grad
                w_param.register_hook(hook_fn)
