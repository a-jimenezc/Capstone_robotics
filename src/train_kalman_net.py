import torch
from torch.utils.data import DataLoader
from src.kalman_net import KalmanNet, KalmanNetDataset
import os

def train_kalman_net(models_folder,
                     model_name,
                     fixed_parameters,
                     delta_t,
                     no_timesteps,
                     dataset_size,
                     device):
    
    # For reproducibility
    torch.manual_seed(42)
    g = torch.Generator()
    g.manual_seed(42)
    
    ckpt_path = os.path.join(models_folder, model_name)

    i_app_range = (-20.0, 150.0)
    dataset = KalmanNetDataset(fixed_parameters,
                dataset_size,
                no_timesteps,
                i_app_range,
                delta_t)
    
    dl = DataLoader(dataset, batch_size=50, shuffle=True, 
                    drop_last=True, generator=g)
    model = KalmanNet(fixed_parameters, delta_t).to(device)

    # Hyperparameters to tune
    gamma = 1e-6
    lr = 1e-4
    epochs = 20
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for vs_noisy_batch, i_app_batch, states_gt_batch, state0_guess_batch in dl:
            vs_noisy = vs_noisy_batch.to(device)
            i_app = i_app_batch.to(device)
            states = states_gt_batch.to(device)
            state0_guess = state0_guess_batch.to(device)

            states_estimated = model(vs_noisy, i_app, state0_guess) # actual shape: (Batch_size, sim_len-1, 10)
            
            mse = (states_estimated - states[:, 1:, :]).pow(2).mean()
            l2 = torch.zeros((), device=device)
            for p in model.parameters():
                l2 = l2 + p.pow(2).sum()

            loss = mse + gamma * l2

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        # Save the latest model
        ckpt = {
            'model_state_dict': model.state_dict(),
            'fixed_parameters': fixed_parameters,
            'delta_t': delta_t,
            'no_timesteps': no_timesteps
        }

        torch.save(ckpt, ckpt_path)
        
        print(f'epoch {epoch}| loss {total/len(dl)}')
        