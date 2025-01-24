#main.py
import os
import time
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from model import LDPCNetwork
from data import CustomDataset, init_parameter, create_random_samples, read_uncor_sample

# Global samples if sampling_type='Read'
GLOBAL_TRAIN_SAMPLES = None
GLOBAL_VALID_SAMPLES = None

def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model_weights(net, args, current_time, best):
    """Example: save learned weights to a text file (optional)."""
    os.makedirs("./Weights", exist_ok=True)
    if best == True:
        path = f"./Weights/{current_time}_{args.filename}_Opt_Weight_Iter{args.iters_max}.txt"
    else:
        path = f"./Weights/{current_time}_{args.filename}_Weight_Iter{args.iters_max}.txt"

    with open(path, "w") as f:
        f.write("\t".join(map(str, args.sharing)) + "\n\n")
        _write_weight(f, net.cn_weight,  "cn_weight")
        _write_weight(f, net.ucn_weight, "ucn_weight")
        _write_weight(f, net.ch_weight,  "ch_weight")

def _write_weight(f, w_param, name_str):
    if w_param is None:
        f.write(f"None\n\n")
        return
    arr = w_param.detach().cpu().numpy()
    f.write(f"{name_str}:\n")
    if arr.ndim == 1:
        line = "\t".join(f"{x:.6f}" for x in arr)
        f.write(line + "\n\n")
    elif arr.ndim == 2:
        for row in arr:
            line = "\t".join(f"{x:.6f}" for x in row)
            f.write(line + "\n")
        f.write("\n")

def run_validation(args, net, code_param, device, rng, epoch):
    """
    Validation step. 
    Measures time, logs val_loss and BER/FER per SNR to both console and file.
    """
    start_time = time.time()
    net.eval()
    num_snr = len(args.SNR_array)

    if args.sampling_type == 'Random':
        llrs_valid = create_random_samples(
            sample_num=args.valid_num,
            code_rate=code_param.code_rate,
            SNR_array=args.SNR_array,
            rng=rng,
            N=code_param.N
        )
    elif args.sampling_type == 'Read':
        global GLOBAL_VALID_SAMPLES
        llrs_valid = GLOBAL_VALID_SAMPLES
        if llrs_valid is None or len(llrs_valid) == 0:
            logging.info("[Validation] No valid samples.")
            return 0.0

    if len(llrs_valid) == 0:
        logging.info("[Validation] Got empty samples.")
        return 0.0

    ds_val = CustomDataset(llrs_valid)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    total_loss = 0.0
    total_count = 0
    bit_err_snr = np.zeros(num_snr, dtype=np.int64)
    frm_err_snr = np.zeros(num_snr, dtype=np.int64)
    samp_snr    = np.zeros(num_snr, dtype=np.int64)
    bit_snr     = np.zeros(num_snr, dtype=np.int64)

    sample_idx = 0
    with torch.no_grad():
        for llr_batch in loader_val:
            llr_batch = llr_batch.to(device)
            loss_val, dec_llr = net(llr_batch, return_dec_llr=True)
            
            if args.systematic == 'on':
                dec_llr = dec_llr[:,:code_param.N-code_param.M]

            total_loss  += loss_val.item() * args.batch_size
            total_count += args.batch_size

            dec_bits = (dec_llr < 0).int().cpu().numpy()
            for b in range(args.batch_size):
                snr_id = sample_idx % num_snr
                sample_idx += 1
                b_err = np.sum(dec_bits[b])
                f_err= (b_err > 0)
                bit_err_snr[snr_id] += b_err
                frm_err_snr[snr_id] += int(f_err)
                bit_snr[snr_id]     += code_param.N
                samp_snr[snr_id]    += 1

    avg_loss = total_loss / total_count if total_count else 0.0
    elapsed  = time.time() - start_time
    logging.info(f"[Epoch {epoch}] valid loss={avg_loss:.4f}, valid time={elapsed:.2f}s")

    # Print SNR-based stats
    for i, snr_val in enumerate(args.SNR_array):
        if samp_snr[i] == 0:
            logging.info(f"SNR={snr_val:.3f} BER=0.00 FER=0.00")
            continue
        ber = bit_err_snr[i]/ bit_snr[i]
        fer = frm_err_snr[i]/ samp_snr[i]
        logging.info(f"SNR={snr_val:.3f} BER={ber:.2e} FER={fer:.2e}")

    return avg_loss

def run_training(args, net, code_param, device, rng, epoch):
    """
    Training step.
    Measures time, logs train_loss to both console and file.
    """
    start_time = time.time()
    net.train()

    if args.sampling_type == 'Random':
        llrs_train = create_random_samples(
            sample_num=args.training_num,
            code_rate=code_param.code_rate,
            SNR_array=args.SNR_array,
            rng=rng,
            N=code_param.N
        )
    elif args.sampling_type == 'Read':
        global GLOBAL_TRAIN_SAMPLES
        llrs_train = GLOBAL_TRAIN_SAMPLES
        if llrs_train is None or len(llrs_train) == 0:
            logging.info("[Training] No train samples.")
            return 0.0

    if len(llrs_train) == 0:
        logging.info("[Training] Got empty samples.")
        return 0.0

    ds_train = CustomDataset(llrs_train)
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learn_rate)

    total_loss = 0.0
    total_count= 0
    for llr_batch in loader_train:
        llr_batch = llr_batch.to(device)
        optimizer.zero_grad()
        loss_val = net(llr_batch)
        loss_val.backward()
        optimizer.step()
        net.clamp_weights()
        bs = llr_batch.size(0)
        total_loss  += loss_val.item() * bs
        total_count += bs

    train_loss = total_loss / total_count if total_count else 0.0
    elapsed = time.time() - start_time
    logging.info(f"[Epoch {epoch}] training loss={train_loss:.4f}, train time={elapsed:.2f}s")
    return train_loss

def load_file_once(args,code_param):
    global GLOBAL_TRAIN_SAMPLES, GLOBAL_VALID_SAMPLES
    file_path = f"./Input/[Uncor]_{args.filename}.txt"
    if args.training_num > 0:
        GLOBAL_TRAIN_SAMPLES = read_uncor_sample(file_path, args.training_num, code_param.N)
    if args.valid_num > 0:
        GLOBAL_VALID_SAMPLES = read_uncor_sample(file_path, args.valid_num, code_param.N,
                                                    start_idx=args.training_num)


def setup_environment(args):
    # Set random seed
    set_seed(args.seed_in)

    # Create directory for weights if it doesn't exist
    os.makedirs("./Weights", exist_ok=True)

    # Generate current time string for unique perf_path
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    perf_path = f"./Weights/{current_time}_{args.filename}_Performance.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(perf_path, mode='w')
        ]
    )

    for k, v in vars(args).items():
        logging.info(f"{k}={v}")
    logging.info("")

    # Prepare device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Initialize RNG
    rng = np.random.RandomState(args.seed_in)

    return current_time, rng, device

def main(args):
    # Setup environment
    current_time, rng, device = setup_environment(args)

    code_param = init_parameter(args.filename, args.z_factor, args.cn_mode)

    # If sampling_type=1, load global samples from file once
    if args.sampling_type == 'Read':
        load_file_once(args,code_param)


    # Build network
    net = LDPCNetwork(code_param, args, device).to(device)
    
    if args.input_weight == "input":
        net.load_init_weights_from_file(args)
        
    if args.fixed_iter > 0:
        net.freeze_first_iters(args)
    
    best_val_loss = float("inf")

    # Main epoch loop
    for epoch in range(1, args.epoch_input + 1):
        val_loss   = run_validation(args, net, code_param, device, rng, epoch)
        train_loss = run_training(args, net, code_param, device, rng, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_weights(net, args, current_time, best=True)
            logging.info("[Info] Best model saved.")

        save_model_weights(net, args, current_time, best=False)
        logging.info(f"Epoch {epoch} done\n")

    logging.info("All done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--filename', type=str, default='wman_N0576_R34_z24')
    parser.add_argument('--z_factor', type=int, default=24)
    parser.add_argument('--clip_llr', type=float, default=20)
    parser.add_argument('--decoding_type', type=str, default='SP', choices=['SP','MS'])
    parser.add_argument('--sharing', nargs='+', type=int, default=[3,0,3]) #[cn_weight,ucn_weight,ch_weight], 1: Edges/Iters 2: Node/Iter 3: Iter, 4: Edge, 5: Node
    parser.add_argument('--iters_max', type=int, default=20)
    parser.add_argument('--fixed_iter', type=int, default=0)
    parser.add_argument('--systematic', type=str, default='off', choices=['off', 'on'])

    parser.add_argument('--init_cn_weight', type=int, default=1)
    parser.add_argument('--init_ch_weight', type=int, default=1)
    parser.add_argument('--input_weight', type=str, default='none', choices=['none','input'])

    parser.add_argument('--cn_mode', type=str, default='parallel',
                        choices=['sequential', 'parallel'],
                        help='Choose the CN update mode: sequential or parallel') 

    parser.add_argument('--loss_option', type=str, default='last', choices = ['multi', 'last'])
    parser.add_argument('--loss_function', type=str, default='FER', choices=['BCE', 'Soft_BER', 'FER'])    

    parser.add_argument('--sampling_type', type=str, default='Random',choices=['Random','Read','Collect']) #'Collect"is not yet implemented.
    parser.add_argument('--SNR_array', nargs='+', type=float, 
                        default=[2,2.5,3.0,3.5,4.0])
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--training_num', type=int, default=10000)
    parser.add_argument('--valid_num', type=int, default=10000)
    parser.add_argument('--epoch_input', type=int, default=100)
    parser.add_argument('--learn_rate', type=float, default=1e-3)
    parser.add_argument('--seed_in', type=int, default=42)
    

    args = parser.parse_args()
    main(args)
