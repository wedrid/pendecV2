from gurobipy import *
import numpy as np

class MistoInteri:
    def __init__(self, A, b, constraint):
        self.__X = A
        self.__Y = b

        #__P e __N?
        self.__P = A.shape[1]  # numero variabili (sarebbe n)
        self.__N = len(b) # numero esempi (è m)
        #print("DIM")
        #print(self.__P)
        #print(self.__N)

        self.__debug = False
        self.__gurobi_limit = False
        self.__time_limit = 60*45 #variabile
        self.__n_cpus = 1
        self.result = self.__gurobi_optimization(constraint)
        #print("Result = " + str(self.result))  


    def __gurobi_optimization(self, k):
            # Creating Gurobi model
            model = Model()
            if not self.__debug:
                # Quieting Gurobi output
                model.setParam("OutputFlag", False)
            if self.__gurobi_limit:
                #model.setParam("TimeLimit", self.__gurobi_time_limit)
                pass
            else:
                model.setParam("TimeLimit", self.__time_limit)
            model.setParam("Threads", self.__n_cpus)
            model.setParam("IntFeasTol", 1e-09)

            # Add variables to the model
            phi, delta = [], []
            for j in range(1, self.__P + 1):
                phi.append(model.addVar(lb=-1000, ub=1000, vtype=GRB.CONTINUOUS, name="phi{}".format(j)))
                delta.append(model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="delta{}".format(j)))

            f = 0
            for i in range(self.__N):
                f += np.dot(self.__Y[i]  - quicksum(phi[j] * self.__X[i, j] for j in range(self.__P)),
                            self.__Y[i]  - quicksum(phi[j] * self.__X[i, j] for j in range(self.__P)))
            model.setObjective(f)

            M = 1e04

            # Add constraints to the model
            for j in range(self.__P):
                model.addConstr(-M * delta[j] <= phi[j])
                model.addConstr(phi[j] <= M * delta[j])
            model.addConstr(quicksum(delta[j] for j in range(self.__P)) <= k)

            # Normal equations
            #if self.__normal_eqs:
            #    e = np.ones(self.__N)
            #    XX = np.matmul(np.transpose(self.__X), self.__X)
            #    XY = np.matmul(np.transpose(self.__X), self.__Y)
            #    for i in range(self.__P):
            #        model.addConstr(np.dot(XX[i], phi) - XY[i] * np.dot(self.__X[:, i], e) <= M * (1 - delta[i]))
            #        model.addConstr(np.dot(XX[i], phi) - XY[i] * np.dot(self.__X[:, i], e) >= -M * (1 - delta[i]))

            # Solve
            model.optimize()

            # Print variables
            if self.__debug:
                for v in model.getVars():
                    print("Var: {}, Value: {}".format(v.varName, v.x))

            return [model.getVarByName("phi{}".format(j)).x for j in range(1, self.__P + 1)], model.getObjective().getValue()
