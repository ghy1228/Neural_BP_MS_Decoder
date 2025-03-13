#main.py
import os
import time
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from model import LDPCNetwork
from data import CustomDataset, init_parameter, create_random_samples
import pdb

# Global samples if sampling_type='Read'
GLOBAL_TRAIN_SAMPLES = {
    "SNR": None, "LLR": None,
}
GLOBAL_VALID_SAMPLES = {
    "SNR": None, "LLR": None,
}

def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model_weights(net, args, best):
    """Example: save learned weights to a text file (optional)."""
    os.makedirs("./Weights", exist_ok=True)
    if best == True:
        path = f"./{args.folder}/{args.out_filename}_Opt_Weight.txt"
    else:
        path = f"./{args.folder}/{args.out_filename}_Weight.txt"

    with open(path, "w") as f:
        f.write("\t".join(map(str, args.sharing)) + "\n\n")
        _write_weight(f, net.cn_weight,  "cn_weight")
        _write_weight(f, net.ucn_weight, "ucn_weight")
        _write_weight(f, net.ch_weight,  "ch_weight")
        _write_weight(f, net.cn_bias,  "cn_bias")
        _write_weight(f, net.ucn_bias,  "ucn_bias")

def _write_weight(f, w_param, name_str):
    if w_param is None:
        f.write(f"None\n\n")
        return
    arr = w_param.detach().cpu().numpy()
    f.write(f"{name_str}:\n")
    
    if arr.ndim == 1:
        for x in arr:
            f.write(f"{x:.6f}\n")  # 한 줄에 하나씩 출력 (column-wise)
        f.write("\n")
    elif arr.ndim == 2:
        for row in arr:
            line = "\t".join(f"{x:.6f}" for x in row)
            f.write(line + "\n")
        f.write("\n")

def process_batch(batch, net, code_param, batch_size, systematic, device):
    """
    Process a single batch and return:
      loss_batch, batch_bit_err, batch_frame_err, batch_size, sum_stop_iter,
      frame_error_flags: a tensor (batch_size,) indicating if a frame had an error.
    """
    loss_val, dec_llr_tensor = net(batch, return_dec_llr=True)  # dec_llr_tensor: (iters_max, batch_size, N_effective)
    iters_max = dec_llr_tensor.size(0)
    if systematic == 'on':
        effective_N = code_param.N - code_param.M
        dec_llr_tensor = dec_llr_tensor[:, :, :effective_N]
    else:
        effective_N = code_param.N

    # Compute decisions
    dec_bits_tensor = (dec_llr_tensor < 0).float()
    H_torch = torch.tensor(code_param.h_matrix.T, dtype=torch.float, device=device)
    syndrome_tensor = torch.matmul(dec_bits_tensor, H_torch) % 2
    valid_mask = (syndrome_tensor == 0).all(dim=2)  # shape: (iters_max, batch_size)
    has_valid = valid_mask.any(dim=0)
    first_valid = torch.where(
        has_valid,
        torch.argmax(valid_mask.float(), dim=0),
        torch.full((batch_size,), iters_max - 1, device=device, dtype=torch.long)
    )
    
    final_dec_llr = dec_llr_tensor[first_valid, torch.arange(batch_size).to(device), :]
    final_dec_bits = (final_dec_llr < 0).float()
    b_err = final_dec_bits.sum(dim=1)
    frame_error_flags = (b_err > 0)  # tensor of bool of shape (batch_size,)
    batch_bit_err = b_err.sum().item()
    batch_frame_err = frame_error_flags.int().sum().item()
    sum_stop_iter = first_valid.float().sum().item()
    loss_batch = loss_val.item() * batch_size

    return loss_batch, batch_bit_err, batch_frame_err, batch_size, sum_stop_iter, frame_error_flags

def load_uncor_samples(args):
    """
    Load samples from file and store in GLOBAL_VALID_SAMPLES and GLOBAL_TRAIN_SAMPLES.
    The first args.valid_num samples go to GLOBAL_VALID_SAMPLES and the next args.training_num samples go to GLOBAL_TRAIN_SAMPLES.
    Each line in the file should have an SNR value followed by LLR values.
    If the total number of samples in the file is less than args.valid_num + args.training_num, exit with an error.
    Stores the SNR and LLR values as numpy arrays.
    """
    global GLOBAL_VALID_SAMPLES, GLOBAL_TRAIN_SAMPLES

    file_path = f"./{args.folder}/{args.uncor_filename}_Uncor.txt"
    samples = []
    with open(file_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            try:
                snr_val = float(tokens[0])
                llr_vals = [float(x) for x in tokens[1:]]
                samples.append((snr_val, llr_vals))
            except ValueError:
                continue

    total_required = args.valid_num + args.training_num
    if len(samples) < total_required:
        raise ValueError(f"Not enough samples in the file: required {total_required}, but got {len(samples)}.")

    valid_samples = samples[:args.valid_num]
    train_samples = samples[args.valid_num: args.valid_num + args.training_num]

    if valid_samples:
        snr_valid, llr_valid = zip(*valid_samples)
        GLOBAL_VALID_SAMPLES["SNR"] = np.array(snr_valid)
        GLOBAL_VALID_SAMPLES["LLR"] = np.array(llr_valid)
    else:
        GLOBAL_VALID_SAMPLES["SNR"] = np.array([])
        GLOBAL_VALID_SAMPLES["LLR"] = np.array([])

    if train_samples:
        snr_train, llr_train = zip(*train_samples)
        GLOBAL_TRAIN_SAMPLES["SNR"] = np.array(snr_train)
        GLOBAL_TRAIN_SAMPLES["LLR"] = np.array(llr_train)
    else:
        GLOBAL_TRAIN_SAMPLES["SNR"] = np.array([])
        GLOBAL_TRAIN_SAMPLES["LLR"] = np.array([])

def collect_uncor_samples(args, net, code_param, rng, device):
    """
    Collect uncorrected samples (frames with errors) for each SNR value.
    Instead of storing them in memory, write each erroneous sample directly to a file.
    After collection, reload the data, interleave the samples from different SNRs, and save the final output.
    """

    raw_file_path = f"./{args.folder}/{args.out_filename}_Raw_Uncor.txt"
    
    # Step 1: Collect erroneous samples and immediately save them to a file
    with open(raw_file_path, "w") as f:
        for snr_val in args.SNR_array:
            collected_count = 0
            start_time = time.time()
            frames_attempted = 0

            while collected_count < args.target_uncor_num:
                # Generate random LLR samples for the given SNR
                llrs_batch = create_random_samples(
                    sample_num=args.batch_size,
                    code_rate=code_param.code_rate,
                    SNR_array=[snr_val],
                    rng=rng,
                    N=code_param.N
                )

                ds_batch = CustomDataset(llrs_batch)
                loader = DataLoader(ds_batch, batch_size=args.batch_size, shuffle=False, drop_last=True)

                for batch in loader:
                    frames_attempted += args.batch_size
                    
                    # Process the batch and get error flags
                    (_, _, _, _, _, frame_error_flags) = process_batch(
                        batch, net, code_param, args.batch_size, args.systematic, device
                    )

                    # Ensure frame_error_flags is on CPU before using NumPy
                    frame_error_flags = frame_error_flags.cpu().numpy()

                    # Iterate through each sample in the batch
                    for i in range(args.batch_size):
                        if frame_error_flags[i]:  # If the sample has an error
                            llr_list = batch[i].tolist() if hasattr(batch[i], "tolist") else list(batch[i])
                            llr_str = " ".join(f"{v:.4f}" for v in llr_list)
                            f.write(f"{snr_val:.3f} {llr_str}\n")
                            collected_count += 1

                            # Print progress update with elapsed time
                            elapsed_time = time.time() - start_time
                            print(f"SNR {snr_val}: {collected_count} uncor samples collected | Elapsed Time: {elapsed_time:.2f}s")

                            # Stop when target number of uncorrected samples is reached
                            if collected_count >= args.target_uncor_num:
                                break
                    if collected_count >= args.target_uncor_num:
                        break
            
            elapsed = time.time() - start_time
            fer = collected_count / frames_attempted if frames_attempted > 0 else 0.0
            logging.info(f"SNR {snr_val:.3f}: Frames Attempted {frames_attempted}, Collected {collected_count}, FER {fer:.2e}, Time {elapsed:.2f}s")

    # Step 2: Reload the saved data, interleave the samples, and save to the final file
    interleaved_file_path = f"./{args.folder}/{args.out_filename}_Uncor.txt"
    with open(raw_file_path, "r") as f:
        lines_by_snr = {snr: [] for snr in args.SNR_array}
        
        # Group samples by SNR
        for line in f:
            snr_val = float(line.split()[0])  # First value is the SNR
            lines_by_snr[snr_val].append(line.strip())

    # Perform round-robin interleaving
    with open(interleaved_file_path, "w") as f:
        for i in range(args.target_uncor_num):
            for snr_val in args.SNR_array:
                if i < len(lines_by_snr[snr_val]):
                    f.write(lines_by_snr[snr_val][i] + "\n")

    print(f"Interleaved samples saved to {interleaved_file_path}")


def run_validation(args, net, code_param, device, rng, epoch):
    """
    Validation step.
    Measures time, logs val_loss and BER/FER per SNR to both console and file.
    Now, for each SNR value:
      - Generates (or reads) batches of LLRs.
      - Processes each batch using early stopping based on syndrome check.
      - Uses vectorized operations for per-frame processing.
      - Stops iteration for a frame at the first iteration where H * dec_bits == 0.
      - Computes BER/FER at the stop iteration and tracks the average stop iteration.
    """
    start_time = time.time()
    net.eval()

    total_loss = 0.0
    total_count = 0  # total number of frames processed overall
    # Print header
    logging.info("\nValidation Results for Epoch {}".format(epoch))
    logging.info("{:<7} {:<8} {:<16} {:<10} {:<10} {:<10} {:<8}".format(   
        "SNR", "Frames", "Uncor_Frames", "BER", "FER", "Loss", "Avg Iter"))
    logging.info("-" *75)

    # Loop over each SNR value.
    for snr_val in args.SNR_array:
        frame_num = 0      # frames processed for current SNR
        frame_err = 0      # number of frames with error for current SNR
        bit_err = 0        # total bit errors for current SNR
        loss_sum = 0.0     # cumulative loss for current SNR
        stop_iter_sum = 0.0  # cumulative sum of stop iterations (0-indexed) for current SNR

        # Process batches until stopping condition is met.
        if args.sampling_type == 'Random':
            # Keep generating batches for current SNR.
            while True:
                # Create one batch for the current SNR.
                llrs_batch = create_random_samples(
                    sample_num=args.batch_size,
                    code_rate=code_param.code_rate,
                    SNR_array=[snr_val],  # only current SNR
                    rng=rng,
                    N=code_param.N
                )
                ds_batch = CustomDataset(llrs_batch)
                loader = DataLoader(ds_batch, batch_size=args.batch_size, shuffle=False, drop_last=True)
                # There will be only one batch in this loader.
                for batch in loader:
                    (loss_batch, batch_bit_err, batch_frame_err, batch_size,
                     sum_stop_iter, _) = process_batch(batch, net, code_param, args.batch_size, args.systematic, device)

                    loss_sum += loss_batch
                    frame_num += batch_size
                    bit_err += batch_bit_err
                    frame_err += batch_frame_err
                    stop_iter_sum += sum_stop_iter

                # Break conditions:
                # If valid_num > 0, stop when total frames exceed valid_num.
                if args.valid_num > 0 and frame_num >= args.valid_num:
                    break
                # If valid_num == 0, stop when frame errors exceed target_uncor_num.
                if args.valid_num == -1 and frame_err >= args.target_uncor_num:
                    break

        elif args.sampling_type == 'Read':
            # GLOBAL_VALID_SAMPLES is now assumed to be a dictionary with SNR keys.
            global GLOBAL_VALID_SAMPLES
            idx = np.where(np.isclose(GLOBAL_VALID_SAMPLES["SNR"], snr_val))[0]
            if len(idx) < args.batch_size:
                logging.info(f"[Read] SNR {snr_val:.3f}: {len(idx)} samples, batch_size={args.batch_size}.")
                continue
            llrs = GLOBAL_VALID_SAMPLES["LLR"][idx]
            ds = CustomDataset(llrs)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=True)
            for batch in loader:
                (loss_batch, batch_bit_err, batch_frame_err, batch_size,
                 sum_stop_iter, _) = process_batch(batch, net, code_param, args.batch_size, args.systematic, device)

                loss_sum += loss_batch
                frame_num += batch_size
                bit_err += batch_bit_err
                frame_err += batch_frame_err
                stop_iter_sum += sum_stop_iter

        # Calculate loss, BER, FER, and average stop iteration for current SNR.
        if frame_num > 0:
            avg_loss_snr = loss_sum / frame_num
            ber = bit_err / (frame_num * (code_param.N - code_param.M) if args.systematic == 'on' else frame_num * code_param.N)
            fer = frame_err / frame_num
            # Convert 0-indexed stop iterations to 1-indexed.
            avg_stop_iter = (stop_iter_sum / frame_num) + 1
        else:
            avg_loss_snr, ber, fer, avg_stop_iter = 0.0, 0.0, 0.0, 0.0

        total_loss += loss_sum
        total_count += frame_num

        logging.info("{:<7.3f} {:<12} {:<10} {:<10.2e} {:<10.2e} {:<13.2e} {:<8.2f}".format(snr_val, frame_num, frame_err, ber, fer, avg_loss_snr, avg_stop_iter))
    avg_loss = total_loss / total_count if total_count else 0.0
    elapsed = time.time() - start_time
    logging.info("-" * 75)
    logging.info("[Epoch {}] valid loss {:.4f}, valid time {:.2f}s".format(epoch, avg_loss, elapsed))

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
        llrs_train = GLOBAL_TRAIN_SAMPLES["LLR"]
        if llrs_train is None or len(llrs_train) == 0:
            logging.info("[Training] No train samples.")
            return 0.0

    if len(llrs_train) == 0:
        logging.info("[Training] Got empty samples.")
        return 0.0

    ds_train = CustomDataset(llrs_train)
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False, drop_last=True)
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


def setup_environment(args):
    os.makedirs(f"./{args.folder}", exist_ok=True)
    # Set random seed
    set_seed(args.seed_in)

    # Create directory for weights if it doesn't exist
    os.makedirs("./Weights", exist_ok=True)

    # Generate current time string for unique perf_path
    perf_path = f"./{args.folder}/{args.out_filename}_Performance.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(perf_path, mode='w')
        ]
    )
    logging.info("<------------- Arguments ------------->")
    for k, v in vars(args).items():
        logging.info(f"{k}={v}")
    logging.info("")

    # Prepare device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info("<========================================>")
    logging.info("")
    # Initialize RNG
    rng = np.random.RandomState(args.seed_in)

    return rng, device

def main(args):
    # Setup environment
    rng, device = setup_environment(args)

    code_param = init_parameter(args.PCM_name, args.z_factor, args.cn_mode)

    # Build network
    net = LDPCNetwork(code_param, args, device).to(device)
    
    if args.input_weight == "input":
        net.load_init_weights_from_file(args)
        
    if args.fixed_iter > 0:
        net.freeze_first_iters(args)
    
    if args.sampling_type == "Read":
        load_uncor_samples(args)

    if args.sampling_type == "Collect":
        collect_uncor_samples(args, net, code_param, rng, device)
        save_model_weights(net, args, best=True)
        return

    best_val_loss = float("inf")
    if args.valid_num > 0 or args.valid_num == -1:
        val_loss = run_validation(args, net, code_param, device, rng, 0)

    # Main epoch loop
    for epoch in range(1, args.epoch_input + 1):
        
        train_loss = run_training(args, net, code_param, device, rng, epoch)
        val_loss   = run_validation(args, net, code_param, device, rng, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_weights(net, args, best=True)
            logging.info("[Info] Best model saved.")

        save_model_weights(net, args, best=False)
        logging.info(f"Epoch {epoch} done\n")

    logging.info("All done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--folder', type=str, default='BCH_63_36')
    parser.add_argument('--PCM_name', type=str, default='BCH_63_36')
    parser.add_argument('--out_filename', type=str, default='NMS_stable_Iter100')
    parser.add_argument('--in_filename', type=str, default='')
    parser.add_argument('--uncor_filename', type=str, default='')

    parser.add_argument('--z_factor', type=int, default=1)
    parser.add_argument('--clip_llr', type=float, default=20)
    parser.add_argument('--decoding_type', type=str, default='MS_stable', choices=['SP','MS','MS_stable'])
    parser.add_argument('--q_bit', type=int, default=0) # 0: Floating operation, >=1: Quantization
    parser.add_argument('--sharing', nargs='+', type=int, default=[1,0,2,0,0]) #[cn_weight,ucn_weight,ch_weight,cn_bias,ucn_bias], 1: Edges/Iters 2: Node/Iter 3: Iter, 4: Edge, 5: Node
    parser.add_argument('--iters_max', type=int, default=100)
    parser.add_argument('--fixed_iter', type=int, default=0)
    parser.add_argument('--systematic', type=str, default='off', choices=['off', 'on'])
    

    parser.add_argument('--init_cn_weight', type=int, default=1) #-1: random values from normal distribution
    parser.add_argument('--init_ch_weight', type=int, default=1)
    parser.add_argument('--init_cn_bias', type=int, default=0)
    parser.add_argument('--input_weight', type=str, default='none', choices=['none','input'])

    parser.add_argument('--cn_mode', type=str, default='parallel', choices=['sequential', 'parallel']) 

    parser.add_argument('--loss_option', type=str, default='last', choices = ['multi', 'last'])
    parser.add_argument('--loss_function', type=str, default='FER', choices=['BCE', 'Soft_BER', 'FER'])    

    parser.add_argument('--sampling_type', type=str, default='Random',choices=['Random','Read','Collect'])
    parser.add_argument('--SNR_array', nargs='+', type=float, 
                        default=[2,3,4,5,6,7,8])
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--training_num', type=int, default=10000)
    parser.add_argument('--valid_num', type=int, default=1000)
    parser.add_argument('--target_uncor_num', type=int, default=100)
    parser.add_argument('--epoch_input', type=int, default=100)
    parser.add_argument('--learn_rate', type=float, default=1e-3)
    parser.add_argument('--seed_in', type=int, default=42)
    

    args = parser.parse_args()
    main(args)
