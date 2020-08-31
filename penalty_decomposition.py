from scipy.optimize import minimize
import math
import copy
from regressione_lineare import * 
import csv
from datetime import datetime


class PenaltyDecomposition: 
    #a sto giro non faccio i metodi statici che mi fanno solo casino

    def __init__(self, fun, tau_zero = None, x_0 = None, epsilon_succession = None, gamma = None, max_iterations = None, l0_constraint = None, alfa = None, name="Noname", save=False):
        self.name = name
        self.resultVal = None
        self.fun = fun
        self.x = []
        self.y = []
        self.epsilon_succession = []
        self.number_of_variables = fun.number_of_x
        self.minPoint = None
        self.minVal = None
        self.outerLoopCondition = 1e-4
        self.innerLoopCondition = 1e-5 

        if l0_constraint is None:
            self.l0_constraint = len(fun.number_of_x) # in pratica corrisponde al non mettere il vincolo..
            #per ora, per prova, metto un constraint che l_0 dev'essere 2
            self.l0_constraint = 2 #TODO da rimuovere nel momento in cui si utilizza senza che siano prove
        else:
            self.l0_constraint = l0_constraint

        if max_iterations is None:
            self.max_iterations = 50
        else:
            self.max_iterations = max_iterations

        if(tau_zero is None):
            self.tau_zero = 10
        else: 
            self.tau_zero = tau_zero
        self.tau = self.tau_zero

        #x_0 è l'x di partenza
        if(x_0 is None):
            print("x_0 is required.")
            return
        else:
            self.x.append(x_0)
            #print("[PENALTY DECOMPOSITION] x_0 is " + str(x_0))
        self.y.append(copy.deepcopy(self.x[0]))

        if(epsilon_succession is None):
            pass
            # TODO generare una successione di epsilon (anche se non deve per forza essere una successione)
        else:
            self.epsilon_succession = epsilon_succession

        if(gamma is None):
            print("Gamma is required")
        else: 
            self.gamma = gamma #TODO gamma deve rispettare il vincolo che dev'essere maggiore o uguale v. paper
        
        if(alfa is None):
            self.alfa = 1.2
        
        self.saveIterationsToCSV = save #se si vogliono salvare le iterazioni su csv
        if self.saveIterationsToCSV:
            self.iterationSaver = np.empty((0, 6))
            date = datetime.now()
            div = "DIV" + str(math.floor(self.number_of_variables/l0_constraint))
            self.currentFileName = "./convergence_data/22ago2020/" + name + "-PDiterations-" + str(self.l0_constraint) + "_of_" + str(self.number_of_variables) + "_dt_" + date.replace(microsecond=0).isoformat() + div + ".csv"
            with open(self.currentFileName, mode="a") as csvFile:
                fieldnames = ['k', 'tau', 'f(u)', 'f(v)', 'q(u,v)', '||x-y||']
                writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                writer.writeheader()
            csvFile.close()

    def start(self):
        min = 100000000000


        k = 0
        epsilon = 0.01
        #while k < self.max_iterations: #DONE cambiare criterio di arresto, distanza |x-y|
        while True: 
            #print("ITERATION: " + str(k))
            u = copy.deepcopy(self.x[k])

            argmin_xQtau = self.fun.getQTauOttimoGivenY(self.tau, self.y[k], np.array(np.ones(self.fun.number_of_x))) #nota: nel caso della regressione lineare il terzo parametro è inutilizzato
            #print("ARGMIN: "+ str(argmin_xQtau))
            min_xQtauk = self.fun.getQTauValue(self.tau, argmin_xQtau, self.y[k])
            
            
            
            #questo è quanto è scritto sul paper, chiarini ha fatto una cosa diversa, la metto sotto
            #if min_xQtauk <=self.gamma:
            #    v = copy.deepcopy(self.y[k])
            #else:
            #    v = copy.deepcopy(self.y[0])
            
            
            
            if min_xQtauk < self.fun.getValueInX(self.x[0]):
                #u = copy.deepcopy(self.x[k])
                v = copy.deepcopy(self.y[k])
            else:
                u = copy.deepcopy(self.x[0])
                v = copy.deepcopy(self.y[0])
            
            
            
            qTauValPrev = self.fun.getQTauValue(self.tau, u, v)
            #while self.fun.getQTauXGradientNorm(self.tau, u, v) > epsilon: #TODO criterio per funzione che decresce di poco 
            #print("\t\t\t\t\t\t\t TAU VALUE: " + str(self.tau))
            while True:
                #primo blocco
                
                u = self.fun.getQTauOttimoGivenY(self.tau, v, np.matrix(np.ones(self.fun.number_of_x)))
                #u = np.matrix(u).transpose()
                #print("u -┐\n" + str(u))
                #print("Q TAU VALUE is " + str(self.fun.getQTauValue(self.tau, u, v)))

                #secondo blocco 
                v = self.fun.getFeasibleYQTauArgminGivenX(self.tau, u, self.l0_constraint)
                v = np.matrix(v).transpose()

                if True:
                    #print("------------- Iteration: " + str(k) + " TAU: "+  str(self.tau))
                    #print("u:\n " + str(u))
                    #print("v:\n " + str(v))
                    fu = self.fun.getValueInX(u)[0,0]
                    fv = self.fun.getValueInX(v)[0,0]
                    quv = self.fun.getQTauValue(self.tau, u, v)[0,0]
                    xlessy = np.linalg.norm(self.x[k] - self.y[k])
                    if False:
                        print("\t\t\t\t\t\t\t\t\t\tf(u) " + str(fu))
                        print("\t\t\t\t\t\t\t\t\t\tf(v) " + str(fv))
                        print("\t\t\t\t\t\t\t\t\t\tq(u,v) " + str(quv))
                        print("\t\t\t\t\t\t\t\t\t\tNORMA DISTANZA X-Y " + str(xlessy))
                    if self.saveIterationsToCSV:
                        temp = np.array([[k, self.tau, fu, fv, quv, xlessy]])
                        self.iterationSaver = np.append(self.iterationSaver, temp, axis=0)


                if self.fun.getValueInX(v) < min:
                    min = self.fun.getValueInX(v)
                    minPoint = v



                #print("v -┐\n" + str(v))
                #print("\t\t\t\t\t\t\t\tNORMA --> " + str(self.fun.getQTauXGradientNorm(self.tau, u, v)))
                #ATTENZIONE, in questa implementazione i vettori delle variabili sono VETTORI COLONNA 

                if abs(qTauValPrev - self.fun.getQTauValue(self.tau, u, v)) < self.innerLoopCondition:
                    break
                else:
                    qTauValPrev = self.fun.getQTauValue(self.tau, u, v)


            if self.saveIterationsToCSV:
                #1 apri file
                with open(self.currentFileName, 'a') as file:
                    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for row in self.iterationSaver:
                        writer.writerow(row)
                #2 reinizializza la matrice
                self.iterationSaver = np.empty((0, 6))
                

            self.tau = self.gamma * self.tau 
            #self.tau = self.alfa * self.tau
            self.x.append(u)
            self.y.append(v)
            #epsilon *= 0.5

            
            k+=1
            if np.linalg.norm(self.x[k] - self.y[k]) < self.outerLoopCondition and True:
                break
            
            
        if False:
            print("[PD] FINISH: \n" + str(self.y[len(self.y)-1]))
            print("MIN --> " + str(min))
            print("[PD] VAL (last): " + str(self.fun.getValueInX(self.y[len(self.y)-1])))
            for point in self.y:
                print("[PD] VAL (all): " + str(self.fun.getValueInX(point)))
            #print(self.y)

        self.resultVal = self.fun.getValueInX(self.y[len(self.y)-1])
        self.resultPoint = self.y[len(self.y)-1]

        self.minPoint = minPoint
        self.minVal = min

