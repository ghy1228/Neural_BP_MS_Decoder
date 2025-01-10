# model.py
import torch
import torch.nn as nn
import math
import pdb

class SignSTE(torch.autograd.Function):
    """
    Sign with a custom backward pass:
      forward: sign(x)
      backward: grad = 2/(1 + exp(-x)) - 1
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = 2.0 / (1.0 + torch.exp(-x)) - 1.0
        return grad_output * grad_x

class LDPCNetwork(nn.Module):
    """
    Simple LDPC decoder example.
      - loss_option: 0 -> BCE, 1 -> soft-BER, 2 -> FER with STE
      - loss_type:   0 -> average all iterations, 1 -> last iteration only
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

        self.init_cn_weight = args.init_cn_weight
        self.init_ch_weight = args.init_ch_weight
        self.iters_max   = args.iters_max
        self.clip_llr    = args.clip_llr
        self.loss_option = args.loss_option
        self.loss_type   = args.loss_type
        self.decoding_type = args.decoding_type
        self.batch_size  = args.batch_size
        self.sharing     = args.sharing

        # Initialize optional learnable weights
        self.cn_weight  = self._init_weight(self.sharing[0], prefix="cn")
        self.ucn_weight = self._init_weight(self.sharing[1], prefix="ucn")
        self.ch_weight  = self._init_weight(self.sharing[2], prefix="ch")

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
            else:
                return None
        elif prefix == "ch":
            if   s_val == 2:
                w = torch.ones(self.iters_max, self.N_proto, device=self.device) * self.init_ch_weight
            elif s_val == 3:
                w = torch.ones(self.iters_max, device=self.device) * self.init_ch_weight
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
        bsz = llr_in.size(0)
        v2c_llr = torch.zeros(bsz, self.E, device=self.device)
        c2v_llr = torch.zeros(bsz, self.E, device=self.device)
        sum_llr = torch.zeros(bsz, self.N, device=self.device)

        dec_llr_list = []
        for it in range(self.iters_max):
            # Optional channel-weight application
            w_ch = self._apply_ch_weight(llr_in, it)

            # VN update
            v2c_llr = self._vn_update(w_ch, v2c_llr, c2v_llr, sum_llr)

            # CN update (SP or MS), either parallel or sequential
            if self.decoding_type == 0:  # SP
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
            c2v_llr = self._apply_cn_weight(c2v_unweighted, it)
            sum_llr = self._compute_sum_llr(c2v_llr, sum_llr)

            # Hard-decision LLR for output
            dec_llr = self._decision(llr_in, c2v_llr)
            dec_llr_list.append(dec_llr)

        loss_val = self._compute_loss_all(dec_llr_list)
        if return_dec_llr:
            return loss_val, dec_llr_list[-1]
        return loss_val

    def _apply_ch_weight(self, llr_in, it_idx):
        """
        Apply channel (ch) weights if needed (e.g. per-iteration, scalar or proto-based).
        """
        if self.ch_weight is None:
            return llr_in
        s_val = self.sharing[2]
        if s_val == 2:
            v = torch.arange(self.N, device=self.device)
            proto_col = v // (self.N // self.N_proto)
            w_iter = self.ch_weight[it_idx, proto_col]
            w_iter = w_iter.unsqueeze(0).expand(llr_in.size(0), -1)
            return llr_in * w_iter
        elif s_val == 3:
            scalar = self.ch_weight[it_idx]
            return llr_in * scalar
        return llr_in

    def _apply_cn_weight(self, c2v_llr, it_idx):
        """
        Apply CN weights if sharing is enabled.
        Clamps final LLR within [-clip_llr, clip_llr].
        """
        if self.cn_weight is None:
            return c2v_llr
        s_val = self.sharing[0]
        if s_val == 1:
            w_edge = self.cn_weight[it_idx, :]   # shape = [E_proto]
            w_edge = torch.repeat_interleave(w_edge, repeats=self.z_factor, dim=0)
            w_edge = w_edge.unsqueeze(0).expand(c2v_llr.size(0), -1)
            return torch.clamp(c2v_llr * w_edge, -self.clip_llr, self.clip_llr)
        elif s_val == 2:
            proto_row = self.edge_to_cn // self.z_factor
            w_m = self.cn_weight[it_idx, proto_row]
            w_m = w_m.unsqueeze(0).expand(c2v_llr.size(0), -1)
            return torch.clamp(c2v_llr * w_m, -self.clip_llr, self.clip_llr)
        elif s_val == 3:
            scalar = self.cn_weight[it_idx]
            return torch.clamp(c2v_llr * scalar, -self.clip_llr, self.clip_llr)
        return c2v_llr

    def _apply_ucn_weight(self, c2v_llr, it_idx):
        """
        (Optional) for extended weighting. Currently not used.
        """
        return c2v_llr

    def _vn_update(self, w_ch, v2c_llr, c2v_llr, sum_llr):
        """
        VN update: v2c_llr = (channel LLR + sum of other c2v) - current c2v.
        Clamps at clip_llr.
        """
        v_idx = self.edge_to_vn
        wch_g = w_ch.gather(1, v_idx.unsqueeze(0).expand(w_ch.size(0), -1))
        sum_g = sum_llr.gather(1, v_idx.unsqueeze(0).expand(sum_llr.size(0), -1))
        out = wch_g + sum_g - c2v_llr
        return torch.clamp(out, -self.clip_llr, self.clip_llr)

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
            out = torch.clamp(out, -self.clip_llr, self.clip_llr)
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
        out = torch.clamp(out, -self.clip_llr, self.clip_llr)

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
            out = torch.clamp(s_j * mg, -self.clip_llr, self.clip_llr)
            c2v_new[:, edges_c] = out

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
        out = torch.clamp(out, -self.clip_llr, self.clip_llr)

        c2v_new = out
        return c2v_new

    def _compute_sum_llr(self, c2v_llr, sum_llr):
        """
        Accumulate CN->VN messages to form sum_llr (used in VN update).
        """
        nxt = torch.zeros_like(sum_llr)
        nxt.index_add_(1, self.edge_to_vn, c2v_llr)
        return nxt

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
        if self.loss_type == 0:
            return torch.stack(losses, dim=0).mean()
        return losses[-1]

    def _compute_loss_one_iter(self, dec_llr):
        """
        Compute one-iteration loss:
          0 -> BCE
          1 -> soft-BER
          2 -> FER (with STE)
        """
        if self.loss_option == 0:  # BCE
            return -torch.log(1.0 - torch.sigmoid(-dec_llr) + 1e-12).mean()
        elif self.loss_option == 1:  # soft-BER
            return torch.sigmoid(-dec_llr).mean()
        elif self.loss_option == 2:  # FER + STE
            min_val = torch.min(dec_llr, dim=1).values
            sign_val = SignSTE.apply(min_val)
            fer = 0.5 * (1.0 - sign_val)
            return fer.mean()
        else:
            return torch.zeros(1, device=dec_llr.device)

    def clamp_weights(self):
        """
        Optimizer step 이후에 각 파라미터를 특정 범위로 clamp.
        """
        if self.cn_weight is not None:
            self.cn_weight.data.clamp_(0.0, 3.0)
        if self.ucn_weight is not None:
            self.ucn_weight.data.clamp_(0.0, 3.0)
        if self.ch_weight is not None:
            self.ch_weight.data.clamp_(0.0, 3.0)
