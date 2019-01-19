# Traffic-Control-System-using-Deep-Q-Learning
An efficient implementation of a traffic signal agent which is modelled as a Convolutional Neural Network and trained using Q-Learning.

Requirements : 
- python v2.7 or v3.5+
- SUMO - Simulation of Urban Mobility (http://www.sumo.dlr.de/userdoc/Downloads.html)
- traci - Traffic Control Interface (http://sumo.dlr.de/wiki/TraCI)
- keras - For Creating CNN Models

Instructions :
Run on terminal:  python tlsClass.py
This creates a .h5 file at the current working directory which is the weight vector which can be initialized to our CNN model.
The training of the model is done using the simple Bellman Equation (Reference : https://joshgreaves.com/reinforcement-learning/understanding-rl-the-bellman-equations/).

Interested programmers to try out Reinforcement Learning Algorithms using OpenAI Gym (https://gym.openai.com/).
