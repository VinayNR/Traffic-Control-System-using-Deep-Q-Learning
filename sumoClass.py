from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import optparse
import random
import traci
import numpy as np

class SumoClass:
    
    def __init__(self, nodes, edges, lanes):
        self.nodes = nodes
        self.edges = edges
        self.lanes = lanes

    def get_options(self):
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
        options, args = optParser.parse_args()
        return options
    
    def get_edges(self):
        edges = traci.edge.getIDList()
        edges = np.asarray(edges)
        return edges[0:7:2]

    def get_lanes(self):
        lanes = []
        edges = self.get_edges()
        for edge in edges:
            for i in range(self.lanes):
                lanes.append(edge + '_' + str(i))
        return lanes
    
    def generate_routefile(self):
        random.seed(42)
        N = 3600
        
        pWE = 1. / 10
        pEW = 1. / 10
        pNS = 1. / 30
        pSN = 1. / 30
        
        pWN = 1. / 20
        pWS = 1. / 20
        pEN = 1. / 20
        pES = 1. / 20
        
        pNW = 1. / 40
        pSW = 1. / 40
        pNE = 1. / 40
        pSE = 1. / 40

        with open("data/cross.rou.xml", "w") as routes:
            print("""<routes>
                <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
                <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="20" guiShape="bus"/>
                
                <route id="right" edges="51o 1i 2o 52i" />
                <route id="left" edges="52o 2i 1o 51i" />
                <route id="down" edges="54o 4i 3o 53i" />
                <route id="up" edges="53o 3i 4o 54i" />
                
                <route id="right_up" edges="51o 1i 4o 54i" />
                <route id="right_down" edges="51o 1i 3o 53i" />
                <route id="left_up" edges="52o 2i 4o 54i" />
                <route id="left_down" edges="52o 2i 3o 53i" />
                
                <route id="up_right" edges="53o 3i 2o 52i" />
                <route id="up_left" edges="53o 3i 1o 51i" />
                <route id="down_right" edges="54o 4i 2o 52i" />
                <route id="down_left" edges="54o 4i 1o 51i" />
                """, file=routes)
            vehNr = 0
            for i in range(N):
                if random.uniform(0, 1) < pWE:
                    print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pEW:
                    print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pNS:
                    print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pSN:
                    print('    <vehicle id="up_%i" type="typeNS" route="up" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                
                if random.uniform(0, 1) < pWN:
                    print('    <vehicle id="left_up_%i" type="typeWE" route="left_up" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pWS:
                    print('    <vehicle id="left_down_%i" type="typeWE" route="left_down" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pEN:
                    print('    <vehicle id="right_up_%i" type="typeWE" route="right_up" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pES:
                    print('    <vehicle id="right_down_%i" type="typeWE" route="right_down" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                
                if random.uniform(0, 1) < pSE:
                    print('    <vehicle id="down_right_%i" type="typeNS" route="down_right" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pSW:
                    print('    <vehicle id="down_left%i" type="typeNS" route="down_left" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pNE:
                    print('    <vehicle id="up_right_%i" type="typeNS" route="up_right" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pNW:
                    print('    <vehicle id="up_left_%i" type="typeNS" route="up_left" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1

            print("</routes>", file=routes)

    def get_state(self):
        
        edges = self.get_edges()
        lanes = self.get_lanes()
        
        cellLength = 7
        offset = 11
        speedLimit = 20
        
        positionMatrix = []
        velocityMatrix = []
        
        for i in range(12):
            positionMatrix.append([])
            velocityMatrix.append([])
            for j in range(12):
                positionMatrix[i].append(0)
                velocityMatrix[i].append(0)
        
        vehicles_road1 = traci.edge.getLastStepVehicleIDs('1i')
        vehicles_road2 = traci.edge.getLastStepVehicleIDs('2i')
        vehicles_road3 = traci.edge.getLastStepVehicleIDs('3i')
        vehicles_road4 = traci.edge.getLastStepVehicleIDs('4i')
        
        junctionPosition = traci.junction.getPosition('0')[0] # (510, 510)
    
        for v in vehicles_road1:
            ind = int(abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
            if(ind < 12):
                positionMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = 1
                velocityMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit

        for v in vehicles_road2:
            ind = int(abs((junctionPosition - traci.vehicle.getPosition(v)[0] + offset)) / cellLength)
            if(ind < 12):
                positionMatrix[3 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[3 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        junctionPosition = traci.junction.getPosition('0')[1]

        for v in vehicles_road3:
            ind = int(abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
            if(ind < 12):
                positionMatrix[8 - traci.vehicle.getLaneIndex(v)][11 - ind] = 1
                velocityMatrix[8 - traci.vehicle.getLaneIndex(v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit
        
        for v in vehicles_road4:
            ind = int(abs((junctionPosition - traci.vehicle.getPosition(v)[1] + offset)) / cellLength)
            if(ind < 12):
                positionMatrix[9 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[9 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        light = []

        if(traci.trafficlight.getPhase('0') == 0):
            light = [1, 0, 0, 0]
        if(traci.trafficlight.getPhase('0') == 1):
            light = [0, 1, 0, 0]
        if(traci.trafficlight.getPhase('0') == 2):
            light = [0 ,0, 1, 0]
        if(traci.trafficlight.getPhase('0') == 3):
            light = [0, 0, 0, 1]
        
        position = np.array(positionMatrix)
        position = position.reshape(1, 12, 12, 1)
        
        velocity = np.array(velocityMatrix)
        velocity = velocity.reshape(1, 12, 12, 1)
        
        lights = np.array(light)
        lights = lights.reshape(1, 4, 1)

        return [position, velocity, lights]

