import math
import copy
from regressione_lineare import * 
from DF_line_search import * 
import random
import csv
from datetime import datetime

class DFPenaltyDecomposition:

    def __init__(self, fun, tau_zero = None, x_0 = None, epsilon_succession = None, gamma = None, max_iterations = None, l0_constraint = None, name="Noname", save=False):
        self.name = name
        self.resultVal = None
        self.fun = fun
        self.x = []
        self.y = []
        self.epsilon_succession = []
        self.number_of_variables = fun.number_of_x
        self.outerLoopCondition = 1e-4

        self.delta = 0.5

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
        
        self.saveIterationsToCSV = save #se si vogliono salvare le iterazioni su csv
        if self.saveIterationsToCSV:
            self.iterationSaver = np.empty((0, 7))
            date = datetime.now()
            div = "DIV" + str(math.floor(self.number_of_variables/l0_constraint))
            self.currentFileName = "./convergence_data/22ago2020/" + name + "-DFPDiterations-" + str(self.l0_constraint) + "_of_" + str(self.number_of_variables) +"_dt_"+ date.replace(microsecond=0).isoformat() + div+".csv"
            with open(self.currentFileName, mode="a") as csvFile:
                fieldnames = ['k', 'tau', 'f(u)', 'f(v)', 'q(u,v)', '||x-y||', 'currentMin']
                writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                writer.writeheader()
            csvFile.close()

    def start(self):
        #the following is one iteration of the outer loop (for k=0,1,..)
        k = 0
        epsilon = 0.01 #TODO provare 10^-4
        min = 100000000000
        #while k < self.max_iterations: 
        while True:
            l = 0
            alfa_tilde = np.ones(self.number_of_variables*2)
            alfa_tilde[0] = 1
            #print(alfa_tilde)

            x_trial = copy.deepcopy(self.x[k])   
            #print(x_trial)         
            #print("Alfa_tilde = " + str(alfa_tilde))
            for i in range(0, self.number_of_variables*2): #nota: in questo modo si prende la prima direzione di discesa disponibile, non è necessariamente detto sia la migliore?
                j = i 
                #j = random.randint(0, self.number_of_variables*2-1)
                alfa_hat = DFLineSearch.lineSearchOnQTau(self.fun, tau = self.tau, d = self.d[j].transpose(), alfa_zero=1, x_in=x_trial, y_in = self.y[k])
                if alfa_hat > epsilon:
                    x_trial = self.x[k] + alfa_hat * self.d[j].transpose()
                    break
            
            #print(x_trial)
            if self.fun.getQTauValue(self.tau, x_trial, self.y[k]) <= self.fun.getValueInX(x_trial):
                u = self.x[k] 
                #u = x_trial
                v = self.y[k]

            else:
                u = self.x[0]
                v = self.y[0]
            
            iteration = 0
            while self.getAlfaTildeMax(alfa_tilde) > epsilon: 
                
                #print(self.getAlfaTildeMax(alfa_tilde))
            #qTauValPrev = self.fun.getQTauValue(self.tau, u, v)
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
                    
                    if alfa_temp == 0: #se la direzione (i-esima) che stiamo provando è in salita, oppure in discesa ma alfa è troppo piccolo, allora rimpicciolisci l'alfa iniziale
                        alfa_tilde[i] = self.delta * alfa_tilde[i]
                    else: #se invece è in discesa e di "abbastanza" il nuovo alfa da cui iniziare la "prossima volta" è quello appena calcolato
                        alfa_tilde[i] = alfa_temp
                    
                    if alfa_temp > epsilon: #se la alfa calcolata è maggiore del threshold epsilon allora aggiorna il punto da cui esploriamo, così che la prossima direzione che esploriamo inizia da li
                        u = u + alfa_temp*self.d[i].transpose()
                    #(altrimenti) u rimane invariato7
                    #print("\t\t\t\t\t\t\t" + str(self.fun.getValueInX(v)))
                    #print(u)
                    #else:
                #print("new u: " + str(u))
                v = self.fun.getFeasibleYQTauArgminGivenX(self.tau, u, self.l0_constraint).transpose() #ERRORE ERA QUA, NON AVEVO MESSO IL TRANSPOSE!!! ATTENZIONEEEEEE
                
                if True:
                
                    print("[DF PD]------------- Iteration: " + str(iteration) + " --k: " + str(k) + " -- tau: " + str(self.tau))
                    #print("u:\n " + str(u))
                    #print("v:\n " + str(v))
                    fu = self.fun.getValueInX(u)
                    fv = self.fun.getValueInX(v)
                    quv = self.fun.getQTauValue(self.tau, u, v)
                    xlessy = np.linalg.norm(self.x[k] - self.y[k])
                    current_min = min
                    print("\t\t\t\t\t\t\t\t\t\tf(u) " + str(fu))
                    print("\t\t\t\t\t\t\t\t\t\tf(v) " + str(fv))
                    print("\t\t\t\t\t\t\t\t\t\tq(u,v) " + str(quv))
                    print("\t\t\t\t\t\t\t\t\t\tNORMA DISTANZA X-Y " + str(xlessy))
                    print("\t\t\t\t\t\t\t\t\t\tCurrent MIN: " + str(min))
                    temp = np.array([[k, self.tau, fu, fv, quv, xlessy, current_min]])
                    if self.saveIterationsToCSV:
                        self.iterationSaver = np.append(self.iterationSaver, temp, axis=0)
                    #print(alfa_tilde)
                iteration+=1
                if self.fun.getValueInX(v) < min:
                    min = self.fun.getValueInX(v)
                    minPoint = v
                if self.fun.getValueInX(v) > min and False:
                    print("!! RISALITA")
                    break
                
            if self.saveIterationsToCSV:
                #1 apri file
                with open(self.currentFileName, 'a') as file:
                    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for row in self.iterationSaver:
                        writer.writerow(row)
                #2 reinizializza la matrice
                self.iterationSaver = np.empty((0, 7))

            self.tau = self.gamma * self.tau 
            #print("Ultimo u: " + str(u))
            #print("Ultimo v: " + str(v))
            self.x.append(u)
            self.y.append(v)

            k+=1
            
            if np.linalg.norm(self.x[k] - self.y[k]) < self.outerLoopCondition and True: #TODO 1e-3 #attenzione, 1e-3 inizia a risalire drasticamente
                #print("Ultimo x: \n" + str(self.x[k]))
                #print("Ultimo y: \n" + str(self.y[k]))
                
                break
            
             #print("K = " + str(k))
            
        if False:
            print("[DF PD] FINISH: \n" + str(self.y[len(self.y)-1]))
            print("[DF PD] VAL: " + str(self.fun.getValueInX(self.y[len(self.y)-1])))
            print("valX_0: " + str(self.fun.getValueInX(self.y[0])))
            print("min " + str(min))
        
        #for point in self.y:
            #print("[DF PD] VAL (all): " + str(self.fun.getValueInX(point)))
        
        
        self.resultVal = self.fun.getValueInX(self.y[len(self.y)-1])
        self.resultPoint = self.y[len(self.y)-1]

        self.minPoint = minPoint
        self.minVal = min


    def getAlfaTildeMax(self, alfa_tilde):
        return np.amax(alfa_tilde)
    

    def startWithBRestart(self):
        if self.saveIterationsToCSV:
            with open(self.currentFileName, 'a') as csvFile:
                    fieldnames = ['k', 'tau', 'f(u)', 'f(v)', 'q(u,v)', '||x-y||', 'currentMin']
                    writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                    writer.writeheader()
        #the following is one iteration of the outer loop (for k=0,1,..)
        k = 0
        epsilon = 0.01 #TODO provare 10^-4
        min = 100000000000
        #while k < self.max_iterations: 
        while True:
            l = 0
            alfa_tilde = np.ones(self.number_of_variables*2)
            alfa_tilde[0] = 1
            #print(alfa_tilde)

            x_trial = copy.deepcopy(self.x[k])   
            #print(x_trial)         
            #print("Alfa_tilde = " + str(alfa_tilde))
            for i in range(0, self.number_of_variables*2): #nota: in questo modo si prende la prima direzione di discesa disponibile, non è necessariamente detto sia la migliore?
                j = i 
                #j = random.randint(0, self.number_of_variables*2-1)
                alfa_hat = DFLineSearch.lineSearchOnQTau(self.fun, tau = self.tau, d = self.d[j].transpose(), alfa_zero=1, x_in=x_trial, y_in = self.y[k])
                if alfa_hat > epsilon:
                    x_trial = self.x[k] + alfa_hat * self.d[j].transpose()
                    break
            
            #print(x_trial)
            if self.fun.getQTauValue(self.tau, x_trial, self.y[k]) <= self.fun.getValueInX(x_trial):
                u = self.x[k] 
                #u = x_trial
                v = self.y[k]

            else:
                u = self.x[0]
                v = self.y[0]
            
            iteration = 0
            while self.getAlfaTildeMax(alfa_tilde) > epsilon: 
                
                #print(self.getAlfaTildeMax(alfa_tilde))
            #qTauValPrev = self.fun.getQTauValue(self.tau, u, v)
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
                    
                    if alfa_temp == 0: #se la direzione (i-esima) che stiamo provando è in salita, oppure in discesa ma alfa è troppo piccolo, allora rimpicciolisci l'alfa iniziale
                        alfa_tilde[i] = self.delta * alfa_tilde[i]
                    else: #se invece è in discesa e di "abbastanza" il nuovo alfa da cui iniziare la "prossima volta" è quello appena calcolato
                        alfa_tilde[i] = alfa_temp
                    
                    if alfa_temp > epsilon: #se la alfa calcolata è maggiore del threshold epsilon allora aggiorna il punto da cui esploriamo, così che la prossima direzione che esploriamo inizia da li
                        u = u + alfa_temp*self.d[i].transpose()
                    #(altrimenti) u rimane invariato
                    #print("\t\t\t\t\t\t\t" + str(self.fun.getValueInX(v)))
                    #print(u)
                    #else:
                #print("new u: " + str(u))
                v = self.fun.getFeasibleYQTauArgminGivenX(self.tau, u, self.l0_constraint).transpose() #ERRORE ERA QUA, NON AVEVO MESSO IL TRANSPOSE!!! ATTENZIONEEEEEE
                
                if True:
                    print("[DF PD]------------- Iteration: " + str(iteration) + " --k: " + str(k) + " -- tau: " + str(self.tau))
                    #print("u:\n " + str(u))
                    #print("v:\n " + str(v))
                    fu = self.fun.getValueInX(u)
                    fv = self.fun.getValueInX(v)
                    quv = self.fun.getQTauValue(self.tau, u, v)
                    xlessy = np.linalg.norm(self.x[k] - self.y[k])
                    current_min = min
                    print("\t\t\t\t\t\t\t\t\t\tf(u) " + str(fu))
                    print("\t\t\t\t\t\t\t\t\t\tf(v) " + str(fv))
                    print("\t\t\t\t\t\t\t\t\t\tq(u,v) " + str(quv))
                    print("\t\t\t\t\t\t\t\t\t\tNORMA DISTANZA X-Y " + str(xlessy))
                    print("\t\t\t\t\t\t\t\t\t\tCurrent MIN: " + str(min))
                    temp = np.array([[k, self.tau, fu, fv, quv, xlessy, current_min]])
                    if self.saveIterationsToCSV:
                        self.iterationSaver = np.append(self.iterationSaver, temp, axis=0)
                    #print(alfa_tilde)
                iteration+=1
                if self.fun.getValueInX(v) < min:
                    min = self.fun.getValueInX(v)
                    minPoint = v
                    
                if self.fun.getValueInX(v) > min and False:
                    print("!! RISALITA --> B-restart")
                    self.tau = 1
                    break
                
            if self.saveIterationsToCSV:
                #1 apri file
                with open(self.currentFileName, 'a') as file:
                    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for row in self.iterationSaver:
                        writer.writerow(row)
                #2 reinizializza la matrice
                self.iterationSaver = np.empty((0, 7))

            self.tau = self.gamma * self.tau 
            #print("Ultimo u: " + str(u))
            #print("Ultimo v: " + str(v))
            self.x.append(u)
            self.y.append(v)

            k+=1
            
            if np.linalg.norm(self.x[k] - self.y[k]) < self.outerLoopCondition and True: #TODO 1e-3 #attenzione, 1e-3 inizia a risalire drasticamente
                #print("Ultimo x: \n" + str(self.x[k]))
                #print("Ultimo y: \n" + str(self.y[k]))
                
                break
            
             #print("K = " + str(k))
            
        if False:
            print("[DF PD] FINISH: \n" + str(self.y[len(self.y)-1]))
            print("[DF PD] VAL: " + str(self.fun.getValueInX(self.y[len(self.y)-1])))
            print("valX_0: " + str(self.fun.getValueInX(self.y[0])))
            print("min " + str(min))
        
        #for point in self.y:
            #print("[DF PD] VAL (all): " + str(self.fun.getValueInX(point)))
        
        
        self.resultVal = self.fun.getValueInX(self.y[len(self.y)-1])
        self.resultPoint = self.y[len(self.y)-1]

        self.minPoint = minPoint
        self.minVal = min
    

    def startWithRandomizedStep(self):
        if self.saveIterationsToCSV:
            with open(self.currentFileName, 'a') as csvFile:
                    fieldnames = ['k', 'tau', 'f(u)', 'f(v)', 'q(u,v)', '||x-y||', 'currentMin']
                    writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                    writer.writeheader()
        #the following is one iteration of the outer loop (for k=0,1,..)
        k = 0
        epsilon = 0.01 #TODO provare 10^-4
        min = 100000000000
        #while k < self.max_iterations: 
        while True:
            l = 0
            alfa_tilde = np.ones(self.number_of_variables*2)
            alfa_tilde[0] = 1
            #print(alfa_tilde)

            x_trial = copy.deepcopy(self.x[k])   
            #print(x_trial)         
            #print("Alfa_tilde = " + str(alfa_tilde))
            for i in range(0, self.number_of_variables*2): #nota: in questo modo si prende la prima direzione di discesa disponibile, non è necessariamente detto sia la migliore?
                j = i 
                #j = random.randint(0, self.number_of_variables*2-1)
                alfa_hat = DFLineSearch.lineSearchOnQTau(self.fun, tau = self.tau, d = self.d[j].transpose(), alfa_zero=1, x_in=x_trial, y_in = self.y[k])
                if alfa_hat > epsilon:
                    x_trial = self.x[k] + alfa_hat * self.d[j].transpose()
                    break
            
            #print(x_trial)
            if self.fun.getQTauValue(self.tau, x_trial, self.y[k]) <= self.fun.getValueInX(x_trial):
                u = self.x[k] 
                #u = x_trial
                v = self.y[k]

            else:
                u = self.x[0]
                v = self.y[0]
            
            iteration = 0
            while self.getAlfaTildeMax(alfa_tilde) > epsilon: 
                
                #print(self.getAlfaTildeMax(alfa_tilde))
            #qTauValPrev = self.fun.getQTauValue(self.tau, u, v)
                #print(alfa_tilde)

                randir = []
                for i in range(0, self.number_of_variables*2):
                    randir.append(i)
                random.shuffle(randir)

                #alfa_temp = np.zeros(self.number_of_variables*2) #TODO to be checked
                alfa_temp=None
                #alfa_temp[0] = 4
                for j in range(0, self.number_of_variables*2): #per tutte le direzioni e antidirezioni cardinali... 
                    i = randir[j]
                    alfa_temp = DFLineSearch.lineSearchOnQTau(self.fun, tau = self.tau, d = self.d[i].transpose(), alfa_zero=alfa_tilde[i], x_in=u, y_in=v)
                    
                    if alfa_temp == 0: #se la direzione (i-esima) che stiamo provando è in salita, oppure in discesa ma alfa è troppo piccolo, allora rimpicciolisci l'alfa iniziale
                        alfa_tilde[i] = self.delta * alfa_tilde[i]
                    else: #se invece è in discesa e di "abbastanza" il nuovo alfa da cui iniziare la "prossima volta" è quello appena calcolato
                        alfa_tilde[i] = alfa_temp
                    
                    if alfa_temp > epsilon: #se la alfa calcolata è maggiore del threshold epsilon allora aggiorna il punto da cui esploriamo, così che la prossima direzione che esploriamo inizia da li
                        u = u + alfa_temp*self.d[i].transpose()
                    #(altrimenti) u rimane invariato
                    #print("\t\t\t\t\t\t\t" + str(self.fun.getValueInX(v)))
                    #print(u)
                    #else:
                #print("new u: " + str(u))
                v = self.fun.getFeasibleYQTauArgminGivenX(self.tau, u, self.l0_constraint).transpose() #ERRORE ERA QUA, NON AVEVO MESSO IL TRANSPOSE!!! ATTENZIONEEEEEE
                
                if True:
                    print("[DF PD]------------- Iteration: " + str(iteration) + " --k: " + str(k) + " -- tau: " + str(self.tau))
                    #print("u:\n " + str(u))
                    #print("v:\n " + str(v))
                    fu = self.fun.getValueInX(u)
                    fv = self.fun.getValueInX(v)
                    quv = self.fun.getQTauValue(self.tau, u, v)
                    xlessy = np.linalg.norm(self.x[k] - self.y[k])
                    current_min = min
                    print("\t\t\t\t\t\t\t\t\t\tf(u) " + str(fu))
                    print("\t\t\t\t\t\t\t\t\t\tf(v) " + str(fv))
                    print("\t\t\t\t\t\t\t\t\t\tq(u,v) " + str(quv))
                    print("\t\t\t\t\t\t\t\t\t\tNORMA DISTANZA X-Y " + str(xlessy))
                    print("\t\t\t\t\t\t\t\t\t\tCurrent MIN: " + str(min))
                    temp = np.array([[k, self.tau, fu, fv, quv, xlessy, current_min]])
                    if self.saveIterationsToCSV:
                        self.iterationSaver = np.append(self.iterationSaver, temp, axis=0)
                    #print(alfa_tilde)
                iteration+=1
                if self.fun.getValueInX(v) < min:
                    min = self.fun.getValueInX(v)
                    minPoint = v
                    
                if self.fun.getValueInX(v) > min and False:
                    print("!! RISALITA --> B-restart")
                    self.tau = 1
                    break
                
            if self.saveIterationsToCSV:
                #1 apri file
                with open(self.currentFileName, 'a') as file:
                    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for row in self.iterationSaver:
                        writer.writerow(row)
                #2 reinizializza la matrice
                self.iterationSaver = np.empty((0, 7))

            self.tau = self.gamma * self.tau 
            #print("Ultimo u: " + str(u))
            #print("Ultimo v: " + str(v))
            self.x.append(u)
            self.y.append(v)

            k+=1
            
            if np.linalg.norm(self.x[k] - self.y[k]) < self.outerLoopCondition and True: #TODO 1e-3 #attenzione, 1e-3 inizia a risalire drasticamente
                #print("Ultimo x: \n" + str(self.x[k]))
                #print("Ultimo y: \n" + str(self.y[k]))
                
                break
            
             #print("K = " + str(k))
            
        if False:
            print("[DF PD] FINISH: \n" + str(self.y[len(self.y)-1]))
            print("[DF PD] VAL: " + str(self.fun.getValueInX(self.y[len(self.y)-1])))
            print("valX_0: " + str(self.fun.getValueInX(self.y[0])))
            print("min " + str(min))
        
        #for point in self.y:
            #print("[DF PD] VAL (all): " + str(self.fun.getValueInX(point)))
        
        
        self.resultVal = self.fun.getValueInX(self.y[len(self.y)-1])
        self.resultPoint = self.y[len(self.y)-1]

        self.minPoint = minPoint
        self.minVal = min