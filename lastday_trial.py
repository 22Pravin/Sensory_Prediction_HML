import csv
import os

import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np

lambda_factor = 1  # Forgetting factor
learning_rate = 0.02  # Learning rate


def iterative_learning_update(previous_torques, desired_angles, current_angles, lambda_factor, learning_rate):
    """
    Update control torques based on the iterative learning control law.
    """
    errors = desired_angles - current_angles
    control_update = learning_rate * errors
    new_torques = lambda_factor * previous_torques + control_update
    return new_torques


def simulate_with_phases_and_viewer(model, data, viewer, actuator_list, num_trials=1000, max_time_steps=1000):
    # Target angles for each joint
    target_angles = np.array([np.pi / 4] * len(actuator_list))

    # Dictionary to store trial number as key and (MSE, resultant angle) as value
    error_result = {}
    data.time = 0
    theta_time_series = {trial: [] for trial in range(150, 161)}
    previous_torques = np.zeros(len(actuator_list))
    current_angles = np.array([data.qpos[i]
                              for i in range(len(actuator_list))])

    for trial in range(1, num_trials + 1):  # Adjusted to include trial num 1000
        if trial <= 400:
            phase = 'BaseLine'
        elif 400 < trial <= 600:
            phase = 'Adaptation'
        elif 600 < trial <= 800:
            phase = 'Washout'
        else:
            phase = 'Readaptation'

        trial_angles = []  # Collect angles for each time step

        print(f"Starting trial {trial} for {phase} Phase")
        for time_step in range(max_time_steps):
            mujoco.mj_step(model, data)
            viewer.sync()
            current_angles = np.array([data.qpos[i]
                                      for i in range(len(actuator_list))])
            data.time += model.opt.timestep

            # Append angles for all time steps within the trial if it's within the desired range
            if 150 <= trial <= 160:
                trial_angles.append(current_angles.tolist())
            # if trial <= 100:
            #     trial_angles.append(current_angles.tolist())

        # Save the collected angles for the trial if it's within the range
        if 150 <= trial <= 160:
            theta_time_series[trial] = trial_angles
        # theta_time_series[trial] = trial_angles

        perturbation = 0
        if phase in ['Adaptation', 'Readaptation']:
            perturbation = -1
        new_torques = iterative_learning_update(previous_torques, target_angles, current_angles,
                                                lambda_factor, learning_rate)
        # new_torques[0] += perturbation
        # data.ctrl[:len(actuator_list)] = new_torques + perturbation
        data.ctrl[0] = new_torques[0] + perturbation
        data.ctrl[1] = new_torques[1]
        previous_torques = new_torques

        error_result[trial] = (
            np.linalg.norm(target_angles - current_angles),
            current_angles.tolist())
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
    # Replace with your own actuator list
    actuator_list = ['torque1', 'torque2']
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance *= 5  # Adjust camera distance
        error_result, theta_time_series = simulate_with_phases_and_viewer(
            model, data, viewer, actuator_list)
        csv_file_path = r'C:\MP_Pravin\Mujoko_Sim\Error_Results.csv'  # Define the path for your CSV file
        write_to_csv(error_result, csv_file_path)
        print(f"Results written to {csv_file_path}")

        # Plotting part remains the same
        trials = list(error_result.keys())
        mse_values = [error_result[trial][0] for trial in trials]
        plt.stem(trials, mse_values, basefmt=' ')
        plt.xlabel('Trial Number')
        plt.ylabel('Norm of Error')
        plt.title('Trial Number vs. Norm of Error')
        plt.grid(True)
        plt.show()

        # Plotting Theta1 and Theta2 time series for the first 50 trials
        # Assuming 1000 time steps
        plot_theta_time_series(theta_time_series, 1000)


if __name__ == "__main__":
    main()
