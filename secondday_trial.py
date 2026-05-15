import csv
import os
import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- STEP 1: The Base Neural Network ---
class FastCognitiveAgent(nn.Module):
    def __init__(self):
        super(FastCognitiveAgent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, error_state):
        return self.fc(error_state)

def iterative_learning_update(previous_torques, desired_angles, current_angles, lambda_factor, learning_rate):
    errors = desired_angles - current_angles
    control_update = learning_rate * errors
    return (lambda_factor * previous_torques) + control_update

def simulate_with_phases_and_viewer(model, data, viewer, actuator_list, num_trials=1000, max_time_steps=1000):
    target_angles = np.array([np.pi / 4] * len(actuator_list))
    error_result = {}
    variance_result = {} # NEW: Track ensemble uncertainty
    
    data.time = 0
    theta_time_series = {trial: [] for trial in range(150, 161)}
    
    previous_torques = np.zeros(len(actuator_list))
    slow_lambda = 1.0  
    slow_lr = 0.05     
    alpha = 0.1 

    # --- BOOTSTRAPPING: Initialize the Ensemble ---
    num_models = 5
    print(f"Initializing Bootstrap Ensemble with {num_models} Independent Agents...")
    ensemble = [FastCognitiveAgent() for _ in range(num_models)]
    optimizers = [optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01) for model in ensemble]
    loss_fn = nn.MSELoss()

    for trial in range(1, num_trials + 1):
        if trial <= 400: phase = 'BaseLine'
        elif 400 < trial <= 600: phase = 'Adaptation'
        elif 600 < trial <= 800: phase = 'Washout'
        else: phase = 'Readaptation'

        trial_angles = []
        if trial % 50 == 0:
            print(f"Running Trial {trial} [{phase}]")

        for time_step in range(max_time_steps):
            mujoco.mj_step(model, data)
            viewer.sync()
            current_angles = np.array([data.qpos[i] for i in range(len(actuator_list))])
            data.time += model.opt.timestep

            if 150 <= trial <= 160:
                trial_angles.append(current_angles.tolist())

        if 150 <= trial <= 160:
            theta_time_series[trial] = trial_angles

        # 1. Calculate Error
        errors = target_angles - current_angles
        error_tensor = torch.FloatTensor(errors)

        # --- BOOTSTRAPPING: Forward Pass (Aggregation) ---
        # Get predictions from all 5 models
        ensemble_outputs = [model(error_tensor) for model in ensemble]
        ensemble_stack = torch.stack(ensemble_outputs) # Shape: (5 models, 2 torques)
        
        # Aggregate: Calculate the Mean torque to apply to the robot
        fast_torque_tensor = torch.mean(ensemble_stack, dim=0)
        fast_torque = fast_torque_tensor.detach().numpy()

        # Calculate Uncertainty: Variance between the 5 models
        model_variance = torch.var(ensemble_stack, dim=0).mean().item()
        variance_result[trial] = model_variance

        # 3. Slow ILC
        slow_torque = iterative_learning_update(previous_torques, target_angles, current_angles, slow_lambda, slow_lr)

        # 4. Integrate Torques
        new_torques = (alpha * fast_torque) + ((1 - alpha) * slow_torque)
        perturbation = -1 if phase in ['Adaptation', 'Readaptation'] else 0
            
        data.ctrl[0] = new_torques[0] + perturbation
        data.ctrl[1] = new_torques[1]
        previous_torques = new_torques  

        error_result[trial] = (np.linalg.norm(errors), current_angles.tolist())

        # --- BOOTSTRAPPING: Backward Pass (Independent Updates) ---
        for i in range(num_models):
            optimizers[i].zero_grad()
            # Each model updates based on its OWN output relative to the global error
            individual_output = ensemble_outputs[i]
            target_torque = torch.FloatTensor(individual_output.detach().numpy() + (0.5 * errors))
            loss = loss_fn(individual_output, target_torque)
            loss.backward()
            optimizers[i].step()

    return error_result, variance_result, theta_time_series

def plot_theta_time_series(theta_time_series, max_time_steps):
    desired_angle = np.pi / 4  

    plt.figure(figsize=(10, 5))
    for trial in range(150, 161):
        angles_array = np.array(theta_time_series[trial])
        plt.plot(range(max_time_steps), angles_array[:, 0], label=f'Theta1 Trial {trial}')
    plt.axhline(y=desired_angle, color='r', linestyle='-', label='Desired Angle')
    plt.xlabel('Time Step')
    plt.ylabel('Theta1 (radians)')
    plt.title('Theta1 Across Time Steps (Trials 150-160)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    for trial in range(150, 161):
        angles_array = np.array(theta_time_series[trial])
        plt.plot(range(max_time_steps), angles_array[:, 1], label=f'Theta2 Trial {trial}', linestyle='--')
    plt.axhline(y=desired_angle, color='r', linestyle='-', label='Desired Angle')
    plt.xlabel('Time Step')
    plt.ylabel('Theta2 (radians)')
    plt.title('Theta2 Across Time Steps (Trials 150-160)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def write_to_csv(error_result, filepath):
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Trial Number', 'MSE', 'Resultant Angles'])
        for trial, (mse, angles) in error_result.items():
            writer.writerow([trial, mse] + angles)

def main():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2R.xml')
    actuator_list = ['torque1', 'torque']  
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance *= 5  
        error_result, variance_result, theta_time_series = simulate_with_phases_and_viewer(
            model, data, viewer, actuator_list)
        
        csv_file_path = 'Error_Results_Ensemble.csv'  
        write_to_csv(error_result, csv_file_path)
        print(f"Results successfully written to {csv_file_path}")

        # --- 1. PLOT THE MAIN LEARNING CURVE ---
        trials = list(error_result.keys())
        mse_values = [error_result[trial][0] for trial in trials]
        
        plt.figure(figsize=(10, 5))
        plt.plot(trials, mse_values, color='black', linewidth=1.5)
        plt.xlabel('Trial Number')
        plt.ylabel('Norm of Error')
        plt.title('Motor Learning Curve: Ensemble Dual-Rate Adaptation')
        plt.axvline(x=400, color='r', linestyle='--', label='Perturbation ON')
        plt.axvline(x=600, color='g', linestyle='--', label='Perturbation OFF')
        plt.axvline(x=800, color='b', linestyle='--', label='Perturbation ON (Savings)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # --- 2. NEW PLOT: ENSEMBLE UNCERTAINTY (VARIANCE) ---
        variance_values = list(variance_result.values())
        
        plt.figure(figsize=(10, 5))
        plt.plot(trials, variance_values, color='purple', alpha=0.8)
        plt.xlabel('Trial Number')
        plt.ylabel('Ensemble Variance (Uncertainty)')
        plt.title('Cognitive Agent Uncertainty Across Trials')
        plt.axvline(x=400, color='r', linestyle=':', alpha=0.5)
        plt.axvline(x=600, color='g', linestyle=':', alpha=0.5)
        plt.axvline(x=800, color='b', linestyle=':', alpha=0.5)
        plt.fill_between(trials, variance_values, color='purple', alpha=0.2)
        plt.grid(True)
        plt.show()

        # --- 3. PLOT THETA 1 & THETA 2 ---
        print("Generating Theta 1 and Theta 2 time series charts...")
        plot_theta_time_series(theta_time_series, 1000)

if __name__ == "__main__":
    main()