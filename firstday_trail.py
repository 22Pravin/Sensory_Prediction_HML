import csv
import os
import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- STEP 1: Define the Fast Process Neural Network ---
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

# --- The Slow Process (Implicit ILC) ---
def iterative_learning_update(previous_torques, desired_angles, current_angles, lambda_factor, learning_rate):
    """
    Update control torques based on the classical iterative learning control law.
    Acts as the Slow Process (Cerebellar implicit learning).
    """
    errors = desired_angles - current_angles
    control_update = learning_rate * errors
    new_torques = lambda_factor * previous_torques + control_update
    return new_torques

def simulate_with_phases_and_viewer(model, data, viewer, actuator_list, num_trials=1000, max_time_steps=1000):
    target_angles = np.array([np.pi / 4] * len(actuator_list))
    error_result = {}
    data.time = 0
    theta_time_series = {trial: [] for trial in range(150, 161)}
    
    # Initialize Slow Process variables
    previous_torques = np.zeros(len(actuator_list))
    slow_lambda = 1.0  # High retention
    slow_lr = 0.05     # Slow learning rate

    # Initialize Fast Process (Neural Network)
    fast_model = FastCognitiveAgent()
    # High learning rate, but high weight_decay to simulate rapid forgetting during Washout
    optimizer = optim.Adam(fast_model.parameters(), lr=0.01, weight_decay=0.01) 
    loss_fn = nn.MSELoss()
    
    # Blending factor (Attention between Fast and Slow processes)
    alpha = 0.2 

    for trial in range(1, num_trials + 1):
        if trial <= 400:
            phase = 'BaseLine'
        elif 400 < trial <= 600:
            phase = 'Adaptation'
        elif 600 < trial <= 800:
            phase = 'Washout'
        else:
            phase = 'Readaptation'

        trial_angles = []

        print(f"Starting trial {trial} for {phase} Phase")
        for time_step in range(max_time_steps):
            mujoco.mj_step(model, data)
            viewer.sync()
            current_angles = np.array([data.qpos[i] for i in range(len(actuator_list))])
            data.time += model.opt.timestep

            if 150 <= trial <= 160:
                trial_angles.append(current_angles.tolist())

        if 150 <= trial <= 160:
            theta_time_series[trial] = trial_angles

        # 1. Calculate Errors
        errors = target_angles - current_angles
        error_tensor = torch.FloatTensor(errors)

        # 2. Get Fast Torque (Explicit Strategy)
        fast_torque_tensor = fast_model(error_tensor)
        fast_torque = fast_torque_tensor.detach().numpy()

        # 3. Get Slow Torque (Implicit Motor Memory)
        slow_torque = iterative_learning_update(previous_torques, target_angles, current_angles, slow_lambda, slow_lr)

        # 4. Integrate Torques
        new_torques = (alpha * fast_torque) + ((1 - alpha) * slow_torque)

        # 5. Apply Perturbation and Update MuJoCo
        perturbation = 0
        if phase in ['Adaptation', 'Readaptation']:
            perturbation = -1
            
        data.ctrl[0] = new_torques[0] + perturbation
        data.ctrl[1] = new_torques[1]
        previous_torques = new_torques  # Save for next trial's slow ILC

        error_result[trial] = (np.linalg.norm(errors), current_angles.tolist())

        # 6. Train the Fast Process (Online Backpropagation)
        # The network learns to output a torque that explicitly minimizes the remaining error
        optimizer.zero_grad()
        # Target = current fast output + a corrective nudge based on task error
        target_torque = torch.FloatTensor(fast_torque + (0.5 * errors))
        loss = loss_fn(fast_torque_tensor, target_torque)
        loss.backward()
        optimizer.step()

    return error_result, theta_time_series

def plot_theta_time_series(theta_time_series, max_time_steps):
    desired_angle = np.pi / 4  # Desired angle in radians

    # Plot for Theta1
    plt.figure(figsize=(10, 5))
    for trial in range(150, 161):
        angles_array = np.array(theta_time_series[trial])
        plt.plot(range(max_time_steps),
                 angles_array[:, 0], label=f'Theta1 Trial {trial}')
    plt.axhline(y=desired_angle, color='r',
                linestyle='-', label='Desired Angle')
    plt.xlabel('Time Step')
    plt.ylabel('Theta1 (radians)')
    plt.title('Theta1 Across Time Steps for First 50 Trials')
    plt.legend()
    plt.show()

    # Plot for Theta2
    plt.figure(figsize=(10, 5))
    for trial in range(150, 161):
        angles_array = np.array(theta_time_series[trial])
        plt.plot(range(max_time_steps),
                 angles_array[:, 1], label=f'Theta2 Trial {trial}', linestyle='--')
    plt.axhline(y=desired_angle, color='r',
                linestyle='-', label='Desired Angle')
    plt.xlabel('Time Step')
    plt.ylabel('Theta2 (radians)')
    plt.title('Theta2 Across Time Steps for Trials')
    plt.legend()
    plt.show()

def write_to_csv(error_result, filepath):
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Trial Number', 'MSE', 'Resultant Angles'])
        for trial, (mse, angles) in error_result.items():
            writer.writerow([trial, mse] + angles)


def main():
    # Construct absolute path to the model relative to this script's location
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2R.xml')
    actuator_list = ['torque1', 'torque']  # Note: matched the XML joint names
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance *= 5  
        error_result, theta_time_series = simulate_with_phases_and_viewer(
            model, data, viewer, actuator_list)
        
        # --- 1. SAVE THE CSV ---
        csv_file_path = 'C:\MP_Pravin\Mujoko_Sim\Error_Results.csv'  
        write_to_csv(error_result, csv_file_path)
        print(f"Results successfully written to {csv_file_path}")

        # --- 2. PLOT THE MAIN LEARNING CURVE ---
        trials = list(error_result.keys())
        mse_values = [error_result[trial][0] for trial in trials]
        
        plt.figure(figsize=(10, 5))
        plt.plot(trials, mse_values, marker='o', markersize=2, linestyle='-')
        plt.xlabel('Trial Number')
        plt.ylabel('Norm of Error')
        plt.title('Motor Learning Curve: Dual-Rate Adaptation')
        
        # Add vertical lines to demarcate phases
        plt.axvline(x=400, color='r', linestyle='--', label='Start Adaptation')
        plt.axvline(x=600, color='g', linestyle='--', label='Start Washout')
        plt.axvline(x=800, color='b', linestyle='--', label='Start Readaptation')
        plt.legend()
        plt.grid(True)
        plt.show()

        # --- 3. PLOT THETA 1 & THETA 2 ---
        print("Generating Theta 1 and Theta 2 time series charts...")
        plot_theta_time_series(theta_time_series, 1000)

if __name__ == "__main__":
    main()