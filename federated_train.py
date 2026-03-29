import time
import os
import shutil
import torch
import copy
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.util import init_ddp, cleanup_ddp

def average_weights(w1, w2):
    w_avg = copy.deepcopy(w1)
    for key in w_avg.keys():
        w_avg[key] = (w1[key] + w2[key]) / 2.0
    return w_avg

if __name__ == '__main__':
    # Parse standard PyTorch CycleGAN arguments
    opt = TrainOptions().parse()
    opt.device = init_ddp()
    
    # Federated Learning configuration
    NUM_ROUNDS = 17
    LOCAL_EPOCHS = 1
    
    print("\n--- Starting Federated CycleGAN Training ---")
    
    base_name = opt.name
    client_a_dir = os.path.join(opt.checkpoints_dir, f"{base_name}_client_A")
    client_b_dir = os.path.join(opt.checkpoints_dir, f"{base_name}_client_B")
    server_dir = os.path.join(opt.checkpoints_dir, f"{base_name}_server")
    os.makedirs(server_dir, exist_ok=True)
    os.makedirs(client_a_dir, exist_ok=True)
    os.makedirs(client_b_dir, exist_ok=True)

    # Check if we need to resume or start fresh
    start_round = 1
    for r in range(1, 101):
        if os.walk(client_a_dir):
            if os.path.exists(os.path.join(client_a_dir, f"round_{r}_net_G_A.pth")):
                start_round = r + 1
            else:
                break

    # Initialize Global Model only if we are starting from Round 1 and no global exists
    if start_round == 1 and not os.path.exists(os.path.join(server_dir, "global_net_G_A.pth")):
        print("\n[Server] Initializing global master model...")
        server_opt = copy.deepcopy(opt)
        server_opt.name = f"{base_name}_server"
        server_opt.isTrain = True
        server_model = create_model(server_opt)
        server_model.setup(server_opt)
        server_model.save_networks('global') # saves global_net_G_A.pth etc in server_dir
    else:
        print(f"\n[Server] Successfully detected previous checkpoints. Resuming from Round {start_round}...")

    def train_client(client_name, dataroot, round_num):
        print(f"\n==================================================")
        print(f"  [Client {client_name}] Starting Round {round_num}")
        print(f"==================================================")
        
        client_opt = copy.deepcopy(opt)
        client_opt.name = f"{base_name}_{client_name}"
        client_opt.dataroot = dataroot
        client_opt.n_epochs = LOCAL_EPOCHS
        client_opt.n_epochs_decay = 0
        client_opt.display_id = -1 # Disable visdom to run smoothly in background
        client_opt.print_freq = 100
        client_opt.save_epoch_freq = LOCAL_EPOCHS # Only save at end of local round
        client_opt.continue_train = True
        client_opt.epoch = 'global' # Tell CycleGAN to load "global_net_*.pth"
        
        # Copy global weights from Server -> Client checkpoints dir
        for net in ['G_A', 'G_B', 'D_A', 'D_B']:
            src = os.path.join(server_dir, f"global_net_{net}.pth")
            dst = os.path.join(opt.checkpoints_dir, client_opt.name, f"global_net_{net}.pth")
            if os.path.exists(src):
                shutil.copy(src, dst)

        dataset = create_dataset(client_opt)
        model = create_model(client_opt)
        model.setup(client_opt) # This automatically loads the 'global' networks!
        
        total_iters = 0
        for epoch in range(1, LOCAL_EPOCHS + 1):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            for i, data in enumerate(dataset):
                iter_start_time = time.time()
                total_iters += client_opt.batch_size
                epoch_iter += client_opt.batch_size
                
                model.set_input(data)
                model.optimize_parameters()

                if total_iters % client_opt.print_freq == 0:
                    t_comp = (time.time() - iter_start_time) / client_opt.batch_size
                    losses = model.get_current_losses()
                    loss_str = " ".join([f"{k}: {v:.3f}" for k, v in losses.items()])
                    print(f"[Client {client_name} | Round {round_num} | Epoch {epoch}] {loss_str}")

                iter_data_time = time.time()

            # End of local epoch update
            model.update_learning_rate()
            print(f"[Client {client_name}] Completed Local Epoch {epoch}")
            
        # At end of local epochs, save weights as round_X
        model.save_networks(f"round_{round_num}")
        print(f"[Client {client_name}] Finished Round {round_num}. Weights saved.")

    for fl_round in range(start_round, NUM_ROUNDS + 1):
        print(f"\n\n{'*'*60}\n          FEDERATED LEARNING ROUND {fl_round}\n{'*'*60}")
        
        # 1. Train Client A
        train_client("client_A", "../federated_dataset/client_A_data", fl_round)
        
        # 2. Train Client B
        train_client("client_B", "../federated_dataset/client_B_data", fl_round)
        
        # 3. Server FedAvg
        print(f"\n[Server] Performing FedAvg for Round {fl_round}...")
        for net in ['G_A', 'G_B', 'D_A', 'D_B']:
            path_A = os.path.join(client_a_dir, f"round_{fl_round}_net_{net}.pth")
            path_B = os.path.join(client_b_dir, f"round_{fl_round}_net_{net}.pth")
            
            w_A = torch.load(path_A, map_location='cpu')
            w_B = torch.load(path_B, map_location='cpu')
            
            w_avg = average_weights(w_A, w_B)
            
            # Save averaged weights back as the new global
            server_path = os.path.join(server_dir, f"global_net_{net}.pth")
            torch.save(w_avg, server_path)
            
        print(f"[Server] Global Model updated successfully by aggregating Client A and Client B weights!")

    print("\n[SUCCESS] Federated Training Completed!")
