import math
import copy
from regressione_lineare import * 
from DF_line_search import * 
import random

class DFPenaltyDecomposition:

    def __init__(self, fun, tau_zero = None, x_0 = None, epsilon_succession = None, gamma = None, max_iterations = None, l0_constraint = None):
        self.fun = fun
        self.x = []
        self.y = []
        self.epsilon_succession = []
        self.number_of_variables = fun.number_of_x

        self.delta = 0.99

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
            tmp = np.ones(self.number_of_variables)
            tmp = np.array([tmp]).transpose()
            self.x.append(tmp)
            print(self.x[0])
            #return
        else:
            self.x.append(x_0)
        self.y.append(copy.deepcopy(self.x[0]))
        print("\t\t\t\t\t\t\t\t\t\t\t\t\t" + str(self.fun.getValueInX(self.y[0])))
        if(epsilon_succession is None):
            pass
            # TODO generare una successione di epsilon (anche se non deve per forza essere una successione)
        else:
            self.epsilon_succession = epsilon_succession

        if(gamma is None):
            print("Gamma is required")
        else: 
            self.gamma = gamma #TODO gamma deve rispettare il vincolo che dev'essere maggiore o uguale v. paper
        
        self.d = []
        for i in range(0, self.number_of_variables):
            dir = np.zeros(self.number_of_variables)
            dir = np.array([dir])
            dir[0][i] = 1
            self.d.append(dir)
            
        for i in range(0, self.number_of_variables):
            dir = np.zeros(self.number_of_variables)
            dir = np.array([dir])
            dir[0][i] = -1
            self.d.append(dir)

    def start(self):
        #the following is one iteration of the outer loop (for k=0,1,..)
        k = 0
        epsilon = 0.01
        while k < self.max_iterations: 
            l = 0
            alfa_tilde = np.ones(self.number_of_variables*2)
            alfa_tilde[0] = 1
            print(alfa_tilde)

            x_trial = copy.deepcopy(self.x[k])   
            #print(x_trial)         
            print("Alfa_tilde = " + str(alfa_tilde))
            for i in range(0, self.number_of_variables*2): #nota: in questo modo si prende la prima direzione di discesa disponibile, non è necessariamente detto sia la migliore?
                j = i 
                #j = random.randint(0, self.number_of_variables*2-1)
                alfa_hat = DFLineSearch.lineSearchOnQTau(self.fun, tau = self.tau, d = self.d[j].transpose(), alfa_zero=1, x_in=x_trial, y_in = self.y[k])
                if alfa_hat > epsilon:
                    x_trial = self.x[k] + alfa_hat * self.d[j].transpose()
                    break
            
            print(x_trial)
            if self.fun.getQTauValue(self.tau, x_trial, self.y[k]) <= self.fun.getValueInX(x_trial):
                u = self.x[k] # TODO non x_trial?
                #u = x_trial
                v = self.y[k]

            else:
                u = self.x[0]
                v = self.y[0]
            
            
            while self.getAlfaTildeMax(alfa_tilde) > epsilon: 
            #while True:
                #print(alfa_tilde)

                #randir = []
                #for i in range(0, self.number_of_variables*2):
                #    randir.append(i)
                #random.shuffle(randir)

                #alfa_temp = np.zeros(self.number_of_variables*2) #TODO to be checked
                alfa_temp=None
                #alfa_temp[0] = 4
                for i in range(0, self.number_of_variables*2): #per tutte le direzioni e antidirezioni cardinali... 
                    #i = randir[j]
                    alfa_temp = DFLineSearch.lineSearchOnQTau(self.fun, tau = self.tau, d = self.d[i].transpose(), alfa_zero=alfa_tilde[i], x_in=u, y_in=v)
                    #print("\t\t\t\t\t\t\t\t\t\t\t\t ALFA TEMP: " + str(alfa_temp))
                    #print("passo: " + str(alfa_temp[i]))
                    #print(b)
                    
                    if alfa_temp == 0: #se la direzione (i-esima) che stiamo provando è in salita, oppure in discesa ma alfa è troppo piccolo, allora rimpicciolisci l'alfa iniziale
                        alfa_tilde[i] = self.delta * alfa_tilde[i]
                    else: #se invece è in discesa e di "abbastanza" il nuovo alfa da cui iniziare la "prossima volta" è quello appena calcolato
                        alfa_tilde[i] = alfa_temp
                    
                    if alfa_temp > epsilon: #se la alfa calcolata è maggiore del threshold epsilon allora aggiorna il punto da cui esploriamo, così che la prossima direzione che esploriamo inizia da li
                        u = u + alfa_temp*self.d[i].transpose()
                    #(altrimenti) u rimane invariato7
                    print("\t\t\t\t\t\t\t" + str(self.fun.getValueInX(v)))
                    #print(u)
                    #else:
                #print("new u: " + str(u))
                v = self.fun.getFeasibleYQTauArgminGivenX(self.tau, u, self.l0_constraint).transpose() #ERRORE ERA QUA, NON AVEVO MESSO IL TRANSPOSE!!! ATTENZIONEEEEEE
                print("\t\t\t\t\t\t\t\t\t\t\t\t\t" + str(self.fun.getValueInX(v)))
            self.tau = self.gamma * self.tau 
            self.x.append(u)
            self.y.append(v)
            k+=1

        print("[DF PD] FINISH: \n" + str(self.y[len(self.y)-1]))
        print("VAL: " + str(self.fun.getValueInX(self.y[len(self.y)-1])))
        print("valX_0: " + str(self.fun.getValueInX(self.y[0])))
    
    def getAlfaTildeMax(self, alfa_tilde):
        return np.amax(alfa_tilde)