This project is an implementation of a Snake Game using Reinforcement Learning techniques.
Files
model.py: Contains the neural network model (Linear_QNet) and the Q-learning trainer (QTrainer) used for training the agent.
game.py: Implements the Snake Game environment (SnakeGameAI), including game logic such as movement, collision detection, and UI updates.
agent.py: Defines the Agent class which interacts with the game environment, makes decisions based on the state, and trains the neural network.
How to Run
To start training the agent, simply run the agent.py file:
sh
python agent.py
Key Concepts
Reinforcement Learning: The agent learns to play the Snake Game by interacting with the environment, receiving rewards for certain actions, and updating its neural network accordingly.
Q-Learning: A specific type of reinforcement learning algorithm used in this project to estimate the optimal action to take in each state.
Dependencies
torch: For building and training the neural network.
pygame: For creating the game environment and UI.
numpy: For numerical operations.