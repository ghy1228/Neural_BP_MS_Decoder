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

# Train
run_main(gpu_id = gpu_id, folder = folder, PCM_name = PCM_name, out_filename = f'NMS_Iter{iters_max}', 
         batch_size = 20, training_num = 10000, valid_num = 1000, epoch_input = 10)

# Evaluation
run_main(gpu_id = gpu_id, folder = folder, PCM_name = PCM_name, in_filename = f'NMS_Iter{iters_max}', out_filename = f'NMS_Iter{iters_max}_eval', 
         batch_size = 1000, input_weight = 'input', training_num = 0, valid_num = -1, target_uncor_num = 100, epoch_input = 0)
