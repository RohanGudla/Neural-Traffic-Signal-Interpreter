# Deep Q-Learning for Traffic Signal Control

This project introduces a framework that employs deep Q-learning reinforcement learning to optimize traffic signal control at intersections, with the objective of enhancing traffic flow and reducing congestion.

The purpose of sharing this code is to provide a valuable starting point for individuals interested in exploring deep reinforcement learning using the SUMO traffic simulator. It's important to note that this code is a simplified version extracted from a master's thesis, serving as a practical resource for your projects.

## Getting Started

To set up and execute this project on your local machine, follow these steps:

1. **Install Anaconda**: Download and install Anaconda, a popular Python distribution, from [Anaconda's website](https://www.anaconda.com/distribution).

2. **Install SUMO**: Download and install the SUMO traffic simulator from [SUMO's official website](https://www.dlr.de/ts/en/desktopdefault.aspx/tabid-9883/16931_read-41000/).

3. **Install TensorFlow-GPU**: Follow [this guide](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc) for installing TensorFlow with GPU support using Anaconda. This step assumes you have an NVIDIA GPU.

   ```bash
   conda create --name tf_gpu
   activate tf_gpu
   conda install tensorflow-gpu
   ```

Software Versions Used:
- Python 3.7
- SUMO traffic simulator 1.2.0
- TensorFlow 2.0

## Running the Algorithm

Once you have completed the setup, you can run the algorithm:

1. **Clone the Repository**: Clone or download this repository to your local machine.

2. **Navigate to the Project Directory**: Open a terminal or Anaconda prompt, navigate to the root folder of the project, and activate your TensorFlow-GPU environment if you haven't already:

   ```bash
   cd path/to/project/folder
   conda activate tf_gpu
   ```

3. **Start Training**: Run the training script by executing the following command:

   ```bash
   python training_main.py
   ```

   This command will initiate the training process for the deep Q-learning agent. You don't need to open the SUMO software separately, as everything is managed within the script. If you want to visualize the training process, set the `gui` parameter in the `training_settings.ini` file to `True`. However, please note that real-time visualization can be slower, and you'll need to close the SUMO-GUI after each episode.

4. **Training Settings**: You can customize various training parameters in the `training_settings.ini` file, such as the number of episodes, batch size, learning rate, and more. Adjust these settings as needed to fine-tune the algorithm's performance.

5. **Testing**: To test a trained agent, you can run the testing script:

   ```bash
   python testing_main.py
   ```

   The testing script will run a single episode of simulation using a trained model and store the results in a folder within the `./model` directory. You can specify the model version to test and other parameters in the `testing_settings.ini` file.

## Code Structure

This project is organized into several Python files, each responsible for different aspects of training and testing:

- `training_main.py`: The main script for training the deep Q-learning agent.
- `testing_main.py`: The main script for testing a trained agent.
- `model.py`: Defines the deep neural network architecture used for Q-learning.
- `memory.py`: Manages the experience replay memory used during training.
- `training_simulation.py`: Handles the simulation environment during training.
- `testing_simulation.py`: Handles the simulation environment during testing.
- `TrafficGenerator.py`: Generates vehicle routes for each episode.
- `visualization.py`: Provides functions for data visualization.
- `utils.py`: Contains utility functions for file management and model loading.

## Configuration Settings

Both training and testing settings are defined in `.ini` files:

- `training_settings.ini`: Contains parameters related to training, such as the number of episodes, batch size, learning rate, and more.
- `testing_settings.ini`: Contains parameters for testing, including the model version to test and simulation settings.

You can customize these settings to adapt the agent's behavior and performance to your specific needs.

## Deep Q-Learning Agent

### Framework

The agent is built on the Q-learning reinforcement learning framework, augmented with a deep neural network for function approximation.

### Environment

The environment represents a 4-way intersection with four incoming lanes and four outgoing lanes per arm. Each arm is 750 meters long. The traffic light system allocates one dedicated traffic light to the left-most lane and shares one traffic light among the other three lanes.

### Traffic Generation

Each episode generates 1000 cars with arrival timing following a Weibull distribution (shape 2). This distribution results in rapid initial arrivals, followed by slower arrivals. Three-quarters of the spawned vehicles go straight, while one-quarter turn either left or right. Vehicles have an equal chance of spawning at the beginning of each arm, ensuring diverse arrival layouts. Vehicle generation is randomized for each episode.

### Agent (Traffic Signal Control System - TLCS)

- **State**: The state space is discretized, with presence cells representing the presence of at least one vehicle. There are 20

 cells per arm, distributed across lanes, resulting in 80 cells for the entire intersection.

- **Action**: The agent selects traffic light phases from four predefined phases, each lasting 10 seconds. When a phase changes, a 4-second yellow phase is activated. The four phases are:
  - North-South Advance: Green for lanes in the north and south arm for right-turn or straight.
  - North-South Left Advance: Green for lanes in the north and south arm for left-turn.
  - East-West Advance: Green for lanes in the east and west arm for right-turn or straight.
  - East-West Left Advance: Green for lanes in the east and west arm for left-turn.

- **Reward**: The agent's reward is determined by the change in cumulative waiting time between actions. Waiting time for a car is defined as the number of seconds it remains stationary (speed=0) since spawning. Cumulative waiting time sums the waiting times of all cars in incoming lanes. Once a car exits an oncoming lane (crosses the intersection), its waiting time is no longer counted. As a result, the agent receives a positive reward when it reduces waiting times.

- **Learning Mechanism**: The agent employs the Q-learning equation *Q(s,a) = reward + gamma • max Q'(s',a')* to update action values. A deep neural network is used to approximate the state-action function. The neural network architecture consists of 80 input neurons (representing the state), five hidden layers with 400 neurons each, and an output layer with four neurons corresponding to the four possible actions. Experience replay is implemented, storing agent experiences and randomly selecting batches of samples for neural network training after action value updates using the Q-learning equation.

## Changelog - New Version (Updated on 12 Jan 2020)

- Training results are now stored in a structured folder format, with results numbered incrementally.
- Introducing a Test Mode for testing created model versions with consistent results.
- Enabling dynamic model creation by specifying the width and depth of the feedforward neural network for each training.
- Neural network training is now performed at the end of each episode, improving overall algorithm efficiency.
- Rewriting the neural network code using Keras and TensorFlow 2.0.
- Incorporating settings files (.ini) for both training and testing.
- Adding a minimum number of samples required in memory to commence training.
- Enhancing code readability.

# Deep Q-Learning for Traffic Signal Control

This project introduces a framework that employs deep Q-learning reinforcement learning to optimize traffic signal control at intersections, with the objective of enhancing traffic flow and reducing congestion.

The purpose of sharing this code is to provide a valuable starting point for individuals interested in exploring deep reinforcement learning using the SUMO traffic simulator. It's important to note that this code is a simplified version extracted from a master's thesis, serving as a practical resource for your projects.

**Author**: Andrea Vidali - University of Milano-Bicocca

## Inspiration and Reference

I drew inspiration and referenced the work of Andrea Vidali to recreate this project for personal use and educational purposes.
