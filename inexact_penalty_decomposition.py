from scipy.optimize import minimize
import math
import copy
from regressione_lineare import * 
from armijo import * 

class InexactPenaltyDecomposition: 
    #a sto giro non faccio i metodi statici che mi fanno solo casino



    def __init__(self, fun, tau_zero = None, x_0 = None, epsilon_succession = None, gamma = None, max_iterations = None, l0_constraint = None):
        self.fun = fun
        self.x = []
        self.y = []
        self.epsilon_succession = []
        self.number_of_variables = fun.number_of_x

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
        
        

    def start(self):
        k = 0
        epsilon = 0.01
        while k < self.max_iterations:
            x_temp = copy.deepcopy(self.x[k])
            y_temp = copy.deepcopy(self.y[k])
            alfa = Armijo.armijoOnQTau(self.fun, tau = self.tau, x_in=x_temp, y_in=y_temp)
            x_trial = x_temp - alfa * self.fun.getQTauXGradient(self.tau, x_temp, y_temp)

            if self.fun.getQTauValue(self.tau, x_trial, y_temp) <= self.fun.getValueInX(self.x[0]):
                u = copy.deepcopy(self.x[k])
                v = copy.deepcopy(self.y[k])
            else:
                u = copy.deepcopy(self.x[0])
                v = copy.deepcopy(self.y[0])

            
            while self.fun.getQTauXGradientNorm(self.tau, u, v) > epsilon:
                #primo blocco
                alfa = Armijo.armijoOnQTau(self.fun, tau = self.tau, x_in=u, y_in=v)
                print("ALFA: " + str(alfa))
                u = u - alfa * self.fun.getQTauXGradient(self.tau, u, v)
                #u = np.matrix(u).transpose()
                #print("u -┐\n" + str(u))
                #print("Q TAU VALUE is " + str(self.fun.getQTauValue(self.tau, u, v)))

                #secondo blocco 
                v = self.fun.getFeasibleYQTauArgminGivenX(self.tau, u, self.l0_constraint)
                v = np.matrix(v).transpose()

                #per capire se sto andando in salita o in discesa
                print(self.fun.getQTauValue(self.tau, u, v))

                #print("v -┐\n" + str(v))
                #print("\t\t\t\t\t\t\t\tNORMA --> " + str(self.fun.getQTauXGradientNorm(self.tau, u, v)))
                #ATTENZIONE, in questa implementazione i vettori delle variabili sono VETTORI COLONNA 

            self.tau = self.gamma * self.tau 
            #self.tau = self.alfa * self.tau
            self.x.append(u)
            self.y.append(v)
            #epsilon *= 0.5

            print("\t\t\t\t\t\t\t TAU VALUE: " + str(self.tau))


            k+=1

        print("[INEXACT] FINISH: \n" + str(self.y[len(self.y)-1]))
        print("VAL: " + str(self.fun.getValueInX(self.y[len(self.y)-1])))
        temp = np.array([[0.05653660635842047, 0.0, -0.2031083583060361, -0.9447476503130903, -0.4078710391440083, 0.0, -0.3890436268275701, -0.638997302470053, -0.7270358330345421, -0.7231540783565926, 1.863556377303855, 0.0, -0.6871298643343594, -0.5590708789547428, 0.0, 0.21798514077189976, 0.43644045042577034, 0.8046240510269675, 0.6743726512283135]]).transpose()
        print("VAL misto interi " + str(self.fun.getValueInX(temp)))
        print(self.y)


