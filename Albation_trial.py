import os
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. The Maskable Neural Network ---
class FastCognitiveAgent(nn.Module):
    def __init__(self):
        super(FastCognitiveAgent, self).__init__()
        self.layer1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 2)

    def forward(self, error_state, ablation_mask=None):
        x = self.layer1(error_state)
        x = self.relu(x)
        
        # Apply the lesion mask to the hidden neurons
        if ablation_mask is not None:
            x = x * ablation_mask
            
        x = self.layer2(x)
        return x

def iterative_learning_update(previous_torques, desired_angles, current_angles, lambda_factor, learning_rate):
    errors = desired_angles - current_angles
    control_update = learning_rate * errors
    return (lambda_factor * previous_torques) + control_update

# --- 2. Headless Simulation Function ---
def simulate_with_phases(model, data, actuator_list, ablation_mask=None, num_trials=1000, max_time_steps=1000):
    target_angles = np.array([np.pi / 4] * len(actuator_list))
    error_result = {}
    
    previous_torques = np.zeros(len(actuator_list))
    slow_lambda = 1.0  
    slow_lr = 0.05     

    fast_model = FastCognitiveAgent()
    optimizer = optim.Adam(fast_model.parameters(), lr=0.01, weight_decay=0.01) 
    loss_fn = nn.MSELoss()
    alpha = 0.1 

    for trial in range(1, num_trials + 1):
        if trial <= 400: phase = 'BaseLine'
        elif 400 < trial <= 600: phase = 'Adaptation'
        elif 600 < trial <= 800: phase = 'Washout'
        else: phase = 'Readaptation'

        for time_step in range(max_time_steps):
            mujoco.mj_step(model, data)
            current_angles = np.array([data.qpos[i] for i in range(len(actuator_list))])

        errors = target_angles - current_angles
        error_tensor = torch.FloatTensor(errors)

        # Forward pass with the ablation mask
        fast_torque_tensor = fast_model(error_tensor, ablation_mask)
        fast_torque = fast_torque_tensor.detach().numpy()

        slow_torque = iterative_learning_update(previous_torques, target_angles, current_angles, slow_lambda, slow_lr)

        new_torques = (alpha * fast_torque) + ((1 - alpha) * slow_torque)

        perturbation = -1 if phase in ['Adaptation', 'Readaptation'] else 0
        data.ctrl[0] = new_torques[0] + perturbation
        data.ctrl[1] = new_torques[1]
        previous_torques = new_torques  

        error_result[trial] = np.linalg.norm(errors)

        optimizer.zero_grad()
        target_torque = torch.FloatTensor(fast_torque + (0.5 * errors))
        loss = loss_fn(fast_torque_tensor, target_torque)
        loss.backward()
        optimizer.step()

    return error_result

# --- 3. The Ablation Study Automation ---
def main():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2R.xml')
    actuator_list = ['torque1', 'torque'] 
    
    results_db = {}
    total_errors = {}

    print("Running Baseline Simulation (No Lesions)...")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    baseline_mask = torch.ones(16)  # All neurons active
    results_db['Baseline'] = simulate_with_phases(model, data, actuator_list, baseline_mask)
    total_errors['Baseline'] = sum(results_db['Baseline'].values())

    # Run 16 separate simulations, lesioning one neuron at a time
    for neuron_idx in range(16):
        print(f"Running Ablation Simulation for Neuron {neuron_idx + 1}/16...")
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        # Create mask and zero out the specific neuron
        mask = torch.ones(16)
        mask[neuron_idx] = 0.0
        
        res = simulate_with_phases(model, data, actuator_list, mask)
        results_db[f'Neuron_{neuron_idx}'] = res
        total_errors[f'Neuron_{neuron_idx}'] = sum(res.values())

    # --- Data Analysis & Plotting ---
    
    # 1. Find the most critical neuron (Highest total error when removed)
    # Exclude baseline from the search
    ablation_only_errors = {k: v for k, v in total_errors.items() if k != 'Baseline'}
    worst_neuron = max(ablation_only_errors, key=ablation_only_errors.get)
    
    print(f"\nStudy Complete.")
    print(f"Most critical node: {worst_neuron} (Highest error spike when lesioned)")

    # CHART 1: Bar Chart of Total Cumulative Error
    plt.figure(figsize=(12, 5))
    labels = list(total_errors.keys())
    values = list(total_errors.values())
    colors = ['green' if l == 'Baseline' else 'red' if l == worst_neuron else 'skyblue' for l in labels]
    
    plt.bar(labels, values, color=colors)
    plt.axhline(y=total_errors['Baseline'], color='g', linestyle='--', label='Baseline Threshold')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Total Cumulative Error (1000 Trials)')
    plt.title('Ablation Study: Impact of Single-Neuron Lesions on System Performance')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # CHART 2: 1000-Trial Learning Curve Comparison
    trials = list(results_db['Baseline'].keys())
    baseline_mse = list(results_db['Baseline'].values())
    worst_mse = list(results_db[worst_neuron].values())

    plt.figure(figsize=(10, 5))
    plt.plot(trials, baseline_mse, label='Baseline (All Neurons Active)', color='green', linewidth=1.5)
    plt.plot(trials, worst_mse, label=f'Lesioned ({worst_neuron} Disabled)', color='red', alpha=0.7, linestyle='--')
    
    plt.axvline(x=400, color='grey', linestyle=':', alpha=0.5)
    plt.axvline(x=600, color='grey', linestyle=':', alpha=0.5)
    plt.axvline(x=800, color='grey', linestyle=':', alpha=0.5)
    
    plt.xlabel('Trial Number')
    plt.ylabel('Norm of Error')
    plt.title('Learning Curve Degradation: Baseline vs. Most Critical Lesion')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()