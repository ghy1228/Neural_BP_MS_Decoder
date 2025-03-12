import subprocess

def run_main(**kwargs):
    args = ["python", "main.py"]
    for key, value in kwargs.items():
        if isinstance(value, list):
            if value:  # Avoid empty lists
                args.extend([f"--{key}"] + list(map(str, value)))
        else:
            args.extend([f"--{key}", str(value)])
    print("Running:", " ".join(args))
    subprocess.run(args)

gpu_id = 0
PCM_name = 'wman_N0576_R34_z24'
folder = 'wman_N0576_R34_z24'
iters_max = 20

# ----------------------------
# **Step 1: Train Model**
# ----------------------------
run_main(
    gpu_id=gpu_id, folder=folder, PCM_name=PCM_name, out_filename='Base',
    input_weight='none', sampling_type='Random', SNR_array=[2.0, 2.5, 3.0, 3.5, 4.0],
    batch_size=100, training_num=10000, valid_num=5000, target_uncor_num=100, epoch_input=50
)

# ---------------------------------------
# **Step 2: Collect Uncorrected Samples**
# ---------------------------------------
run_main(
    gpu_id=gpu_id, folder=folder, PCM_name=PCM_name, out_filename='Base_collect',
    input_weight='input', in_filename='Base',iters_max = iters_max,
    sampling_type='Collect', SNR_array=[3.5],
    batch_size=1000, target_uncor_num=10000, epoch_input=3
)

# ----------------------------
# **Step 3: Post Decoder**
# ----------------------------
run_main(
    gpu_id=gpu_id, folder=folder, PCM_name=PCM_name, out_filename='Post',
    input_weight='input', in_filename='Base', uncor_filename = 'Base_collect', iters_max = 30, fixed_iter = 20,
    sampling_type='Read', SNR_array=[3.5],
    batch_size=100, training_num=5000, valid_num=5000, epoch_input=50
)

# ----------------------------
# **Step 4: Evaluation**
# ----------------------------
run_main(
    gpu_id=gpu_id, folder=folder, PCM_name=PCM_name, out_filename='Base_eval',
    input_weight='input', in_filename='Base',  iters_max = 20,
    sampling_type='Random', SNR_array=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
    batch_size=1000, training_num=0, valid_num=-1, target_uncor_num=100, epoch_input=0
)

run_main(
    gpu_id=gpu_id, folder=folder, PCM_name=PCM_name, out_filename='Post_eval',
    input_weight='input', in_filename='Post',  iters_max = 30,
    sampling_type='Random', SNR_array=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
    batch_size=1000, training_num=0, valid_num=-1, target_uncor_num=100, epoch_input=0
)

print("\n All Steps Completed Successfully")
