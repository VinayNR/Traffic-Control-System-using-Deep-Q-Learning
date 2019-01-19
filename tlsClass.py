from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import random
import numpy as np
from collections import deque

from sumolib import checkBinary
import traci

import tensorflow as tf
import keras
from keras.layers import Dense, Input, Conv2D, Flatten
from keras.models import Model

from sumoClass import SumoClass

class TLSClass:
    
    def __init__(self):
        self.gamma = 0.95
        self.epsilon = 0.1
        self.learning_rate = 0.001
        self.memory = deque(maxlen=4000)
        self.model = self.build_model()
        self.action_size = 4

    def build_model(self):
        inputs1 = Input(shape=(12,12,1))
        x1 = Conv2D(16, (4,4), strides=(2,2), activation='relu')(inputs1)
        x1 = Conv2D(32, (2,2), strides=(1,1), activation='relu')(x1)
        x1 = Flatten()(x1)

        inputs2 = Input(shape=(12,12,1))
        x2 = Conv2D(16, (4,4), strides=(2,2), activation='relu')(inputs2)
        x2 = Conv2D(32, (2,2), strides=(1,1), activation='relu')(x2)
        x2 = Flatten()(x2)

        inputs3 = Input(shape=(4, 1))
        x3 = Flatten()(inputs3)

        x = keras.layers.concatenate([x1, x2, x3])
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(4, activation='linear')(x)
        
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=[x])
        model.compile(optimizer=keras.optimizers.RMSprop(lr=self.learning_rate), loss='mse')
                                                         
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    
    nodes = 4
    edges = 4
    lanes = 3
    
    total_queue_length = []
    total_waiting_time = []
    total_fuel_consumption = []
    
    sumoObject = SumoClass(nodes, edges, lanes)
    options = sumoObject.get_options()

    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    
    # To be removed
    #sumoBinary = checkBinary('sumo')
    
    traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])

    sumoObject.generate_routefile()

    num_episodes = 50
    batch_size = 32
    tlsObject = TLSClass()
    
    edges = sumoObject.get_edges()
    lanes = sumoObject.get_lanes()

    # Pre-load the weights
    '''try:
        tlsObject.load('trained_kernel_initialiser.h5')
    except:
        print("No models found to instantiate kernel")
    '''
    traci.close()

    for episode in range(num_episodes):
        print(episode)
        stepz = 0
        reward1 = 0
        reward2 = 0
        queue_length = 0
        waiting_time = 0
        fuel_consumption = 0
        traci.start([sumoBinary, "-c", "data/cross.sumocfg", "-W", "--tripinfo-output", "tripinfo.xml"])
        traci.trafficlight.setPhase("0", 1)
        traci.trafficlight.setPhaseDuration("0", 200)

        while traci.simulation.getMinExpectedNumber() > 0 and stepz < 5000:   # as long as vehicles are there
            traci.simulationStep()
            state = sumoObject.get_state()
            action = tlsObject.act(state)
            light = np.argmax(state[2].reshape(4,))
            
            if(action == 0):
                if(light == 0):
                    reward1 = traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                    reward2 = traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                    for i in range(30):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 0)
                        reward1 += traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                        reward2 += traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
        
                if(light == 1):
                    for i in range(2):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 6)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()

                    # Action Execution
                    reward1 = traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                    reward2 = traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                    for i in range(30):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 0)
                        traci.trafficlight.setPhaseDuration("0", 50)
                        reward1 += traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                        reward2 += traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                    
                if(light == 2):
                    for i in range(2):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 7)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                    
                    # Action Execution
                    reward1 = traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                    reward2 = traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                    for i in range(30):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 0)
                        traci.trafficlight.setPhaseDuration("0", 50)
                        reward1 += traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                        reward2 += traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                
                if(light == 3):
                    for i in range(2):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 7)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                    
                    # Action Execution
                    reward1 = traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                    reward2 = traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                    for i in range(30):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 0)
                        traci.trafficlight.setPhaseDuration("0", 50)
                        reward1 += traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                        reward2 += traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                            
            if(action == 1):
                if(light == 1):
                    reward1 = traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                    reward2 = traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                    for i in range(30):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 1)
                        traci.trafficlight.setPhaseDuration("0", 50)
                        reward1 += traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                        reward2 += traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()

                if(light == 0):
                    for i in range(2):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 7)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()

                    # Action Execution
                    reward1 = traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                    reward2 = traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                    for i in range(30):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 1)
                        traci.trafficlight.setPhaseDuration("0", 50)
                        reward1 += traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                        reward2 += traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()

                if(light == 2):
                    for i in range(2):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 6)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                
                        # Action Execution
                        reward1 = traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                        reward2 = traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                        for i in range(30):
                            stepz += 1
                            traci.trafficlight.setPhase('0', 1)
                            traci.trafficlight.setPhaseDuration("0", 50)
                            reward1 += traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                            reward2 += traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                            queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                            waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                            fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                            traci.simulationStep()
                
                if(light == 3):
                    for i in range(2):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 6)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                        
                        # Action Execution
                        reward1 = traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                        reward2 = traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                        for i in range(30):
                            stepz += 1
                            traci.trafficlight.setPhase('0', 1)
                            traci.trafficlight.setPhaseDuration("0", 50)
                            reward1 += traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                            reward2 += traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                            queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                            waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                            fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                            traci.simulationStep()
                            
            if(action == 2):
                if(light == 0 or light == 1):
                    for i in range(2):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 6)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                    
                    # Action Execution
                    reward1 = traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                    reward2 = traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                    for i in range(30):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 0)
                        traci.trafficlight.setPhaseDuration("0", 50)
                        reward1 += traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                        reward2 += traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()

            if(action == 3):
                if(light == 0 or light == 1):
                    for i in range(2):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 7)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                    
                    # Action Execution
                    reward1 = traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                    reward2 = traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                    for i in range(30):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 1)
                        traci.trafficlight.setPhaseDuration("0", 50)
                        reward1 += traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                        reward2 += traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()

            new_state = sumoObject.get_state()
            reward = reward1 - reward2
            tlsObject.remember(state, action, reward, new_state, False)
            if(len(tlsObject.memory) > batch_size):
                tlsObject.replay(batch_size)

        mem = tlsObject.memory[-1]
        del tlsObject.memory[-1]
        tlsObject.memory.append((mem[0], mem[1], reward, mem[3], True))
        
        #print('Episode - ' + str(episode) + ' Total waiting time - ' + str(queue_length))
        total_queue_length.append(str(episode) + ' : ' + str(queue_length) + ' , ' + str(waiting_time) + ' , ' + str(fuel_consumption))
        traci.close(wait=False)

    for queue_length in total_queue_length:
        print(queue_length)

    # Saving weights
    tlsObject.save('trained_kernel_initialiser_2.h5')

    sys.stdout.flush()
