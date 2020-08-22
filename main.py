from regressione_lineare import *
import numpy as np
from penalty_decomposition import *
from armijo import *
from inexact_penalty_decomposition import * 
from DF_line_search import * 
from Dataset import * 
from DF_penalty_decomposition import * 
import math
from misto_interi import * 
from datetime import datetime
import json
import time
from quadratic_test_problem import *
from exponential_test_problem import *

divisore_constraint = 4
results = {}
currentFileName = ""


def main():
    #quadratic = QuadraticTestProblem()
    #fun = RegressioneLineare(np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]), np.array([[1],[2],[3],[4],[5]]))

    #DFLineSearch.provaLineSearch(None, alfa_zero = 1)
    #dfpd = DFPenaltyDecomposition(fun, x_0=np.array([[1],[2],[3]]), gamma=1.1, max_iterations=20, l0_constraint=2, tau_zero=2)
    #dfpd.start()
    
    #print(fun.getValueInX(np.array([[1],[2],[3]])))
    #print(fun.getValueOfGradientInX(np.array([[1],[2],[3]])))
    
    #print(fun.getQTauXGradientNorm(5, np.array([[1],[2],[3]]), np.array([[1],[2],[3]])))
    #print(fun.getQTauXGradient(5, np.array([[1],[2],[3]]), np.array([[1],[2],[3]])))
    #print(fun.getFeasibleYQTauArgminGivenX(5, np.array([[1],[2],[3]]), 2))
    #runOnCrime()
    #runOnServoDataset()
    if True:
        date = datetime.now()
        global currentFileName 
        currentFileName = "results_div_4-" + date.isoformat() + ".json"
        print("--------------------> filename: " + currentFileName)
        with open(currentFileName, 'w') as outfile:
            json.dump(results, outfile, indent=4)

        runOnServoDataset()
        #runOnHousing()
        #runOnForestFires() #inexact slower, DF only one iteration -> 275.89008406564056 (Exact --> 262)
        #runOnBreastCancer()
        #runOnAutoMPG()
        #runOnAutomobile()
        #runOnCrime() rompe tutto :(
        if False:
            currentFileName = "results_div_2-" + date.isoformat() + ".json"
            with open(currentFileName, 'w') as outfile:
                json.dump(results, outfile, indent=4)
            
            global divisore_constraint
            divisore_constraint = 2
            runOnServoDataset()
            runOnHousing()
            runOnForestFires() #inexact slower, DF only one iteration -> 275.89008406564056 (Exact --> 262)
            runOnBreastCancer()
            
            
            runOnAutoMPG()
            runOnAutomobile()
            #runOnCrime()


    #print(fun.getQTauOttimoGivenY(3, np.array([[1],[2],[3]]), np.matrix([1,2,3])))

def runOnSolarFlares(): #doesn't work
    data = Dataset(name = "solar-flare", directory="./datasets/")
    run(data, 'solar-flare')


def runOnAutoMPG():
    data = Dataset(name = "auto-mpg", directory="./datasets/")
    run(data, 'auto-mpg')

def runOnAutomobile():
    data = Dataset(name="automobile", directory="./datasets/")
    run(data, 'automobile')

def runOnForestFires():
    data = Dataset(name="forest-fires", directory="./datasets/")
    run(data, 'forest-fires')


def runOnServoDataset():
    data = Dataset(name="servo", directory="./datasets/")
    run(data, 'servo')
    
    
    
    if(False):
        X, Y = data.get_dataset()
        Y = np.array([Y])
        Y = Y.transpose()
        print("Shape X " + str(X.shape))
        print("Shape Y " + str(Y.shape))

        fun = RegressioneLineare(X, Y)
        pendec = PenaltyDecomposition(fun, x_0= np.array([X[0]]).transpose(), gamma=1.1, max_iterations=5, l0_constraint=15, tau_zero=1)
        pendec.start()

        inexact = InexactPenaltyDecomposition(fun, x_0=np.array([X[0]]).transpose(), gamma=1.1, max_iterations=5, l0_constraint=15, tau_zero=1)
        inexact.start()

        dfpd = DFPenaltyDecomposition(fun, x_0=np.array([X[0]]).transpose(), gamma=1, max_iterations=1, l0_constraint=15, tau_zero=2)


        dfpd = DFPenaltyDecomposition(fun, x_0 = np.array([X[0]]).transpose(), gamma=1.1, max_iterations=3, l0_constraint=15, tau_zero=1)
        dfpd = DFPenaltyDecomposition(fun, x_0 = np.array([np.ones(fun.number_of_x)]).transpose(), gamma=1.1, max_iterations=3, l0_constraint=15, tau_zero=1)
        dfpd = DFPenaltyDecomposition(fun, x_0 = x0, gamma=1.1, max_iterations=1, l0_constraint=15, tau_zero=1)
        dfpd.start()


def runOnSmallLinearRegression():
    fun = RegressioneLineare(np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]), np.array([[1],[2],[3],[4],[5]]))
    pendec = PenaltyDecomposition(fun, x_0=np.array([[1],[2],[3]]), gamma=1.1, max_iterations=2, l0_constraint=2, tau_zero=2)
    
    #pendec = PenaltyDecomposition(fun, x_0= X[0], gamma=1.2, max_iterations=5, l0_constraint=2, tau_zero=5)
    pendec.start()

    inexact = InexactPenaltyDecomposition(fun, x_0=np.array([[1],[2],[3]]), gamma=1.1, max_iterations=5, l0_constraint=2, tau_zero=2)
    inexact.start()

    dfpd = DFPenaltyDecomposition(fun, x_0 = np.array([[1],[2],[3]]), gamma=1.1, max_iterations=3, l0_constraint=2, tau_zero=2)
    dfpd.start()

def runOnCrime():
    data = Dataset(name="crime", directory="./datasets/")
    #run(data, 'crime')
    X, Y = data.get_dataset()
    Y = np.array([Y])
    Y = Y.transpose()
    name = "Pippo"
    salva = True
    #il seguente è per mettere un tetto massimo a tau, perchè con crime si rompe tutto 
    if False:
        fun = RegressioneLineare(X, Y)
        constraint1 = math.floor(fun.number_of_x / divisore_constraint)
        pendec = PenaltyDecomposition(fun, x_0 = np.array([np.zeros(fun.number_of_x)]).transpose(), gamma=1.1, max_iterations=500000000000000, l0_constraint=constraint1, tau_zero=1, name=name, save=salva)
        pendec.start()
        inexact = InexactPenaltyDecomposition(fun, x_0 = np.array([np.zeros(fun.number_of_x)]).transpose(), gamma=1.1, max_iterations=500000000000000, l0_constraint=constraint1, tau_zero=1, name=name, save=salva)
        inexact.start()
        dfpd = DFPenaltyDecomposition(fun, x_0 = np.array([np.zeros(fun.number_of_x)]).transpose(), gamma=1.1, max_iterations=500000000000000, l0_constraint=constraint1, tau_zero=1, name=name, save=salva)
        #dfpd.start()

def runOnHousing():
    data = Dataset(name="housing", directory="./datasets/")
    run(data, 'housing')

def runOnBreastCancer():
    data = Dataset(name="breast-cancer", directory="./datasets/")
    run(data, 'breast-cancer')

def run(data, name):
    #runTests(data)
    salva = False #per salvare i dati sulla convergenza
    if True: #(uncomment to save results to json) --> no, è ridondante  
        print("FILENAME " + currentFileName)
        X, Y = data.get_dataset()
        Y = np.array([Y])
        Y = Y.transpose()
        print("Shape X " + str(X.shape))
        print("Shape Y " + str(Y.shape)) 
        res_temp = {'name': name, 'shape-x': X.shape, 'shape-y': Y.shape}

        fun = RegressioneLineare(X, Y)
        constraint1 = math.floor(fun.number_of_x / divisore_constraint)
        #TODO iniziare da zero
        pendec = PenaltyDecomposition(fun, x_0 = np.array([np.zeros(fun.number_of_x)]).transpose(), gamma=1.1, max_iterations=500000000000000, l0_constraint=constraint1, tau_zero=1, name=name, save=salva)
        print("[Penalty decomposition on " + name + "] starting at " + time.asctime())
        start = time.time()
        pendec.start()
        end = time.time()
        elapsed = end-start
        #if False:
        res = list(np.array(pendec.resultPoint.transpose())[0] )
        print("------> " + str(res))
        minp = list(  np.array(pendec.minPoint.transpose())[0] )
        print(minp)
        print(pendec.minVal[0,0])
        print(pendec.resultVal[0,0])


        res_temp['pd'] = {'elapsed-time': elapsed, 'return-point': res, 'min-point': minp, 'return-val': pendec.resultVal[0,0], 'min-val': pendec.minVal[0,0],
            'constraint': constraint1
        }



        inexact = InexactPenaltyDecomposition(fun, x_0 = np.array([np.zeros(fun.number_of_x)]).transpose(), gamma=1.1, max_iterations=500000000000000, l0_constraint=constraint1, tau_zero=1, name=name, save=salva)
        print("[Inexact penalty decomposition on " + name + "] starting at " + time.asctime())
        start = time.time()
        inexact.start()
        end = time.time()
        elapsed = end-start

        res = list(np.array(inexact.resultPoint.transpose())[0] )
        print("------> " + str(res))
        minp = list(  np.array(inexact.minPoint.transpose())[0] )
        print(minp)
        print(inexact.minVal[0,0])
        print(inexact.resultVal[0,0])

        
        res_temp['inexact-pd'] = {'elapsed-time': elapsed, 'return-point': res, 'min-point': minp, 'return-val': inexact.resultVal[0,0], 'min-val': inexact.minVal[0,0],
            'constraint': constraint1
        }

        

        dfpd = DFPenaltyDecomposition(fun, x_0 = np.array([np.zeros(fun.number_of_x)]).transpose(), gamma=1.1, max_iterations=500000000000000, l0_constraint=constraint1, tau_zero=1, name=name, save=salva)
        print("[DP penalty decomposition on " + name + "] starting at " + time.asctime())
        start = time.time()
        dfpd.start()
        end = time.time()
        elapsed = end-start

        res = list(np.array(dfpd.resultPoint.transpose())[0] )
        print("------> " + str(res))
        minp = list(  np.array(dfpd.minPoint.transpose())[0] )
        print(minp)
        print(dfpd.minVal)
        print(dfpd.resultVal)

        res_temp['dfpd'] = {'elapsed-time': elapsed, 'return-point': res, 'min-point': minp, 'return-val': dfpd.resultVal, 'min-val': dfpd.minVal,
            'constraint': constraint1
        }

        dfpd = DFPenaltyDecomposition(fun, x_0 = np.array([np.zeros(fun.number_of_x)]).transpose(), gamma=1.1, max_iterations=500000000000000, l0_constraint=constraint1, tau_zero=1, name=name, save=salva)
        print("[DP penalty decomposition on " + name + "] starting at " + time.asctime())
        start = time.time()
        dfpd.startWithBRestart()
        end = time.time()
        elapsed = end-start

        res = list(np.array(dfpd.resultPoint.transpose())[0] )
        print("------> " + str(res))
        minp = list(  np.array(dfpd.minPoint.transpose())[0] )
        print(minp)
        print("Min value, B restart: " + str(dfpd.minVal))
        print("Res value " + str(dfpd.resultVal))

        res_temp['dfpd_b_restart'] = {'elapsed-time': elapsed, 'return-point': res, 'min-point': minp, 'return-val': dfpd.resultVal, 'min-val': dfpd.minVal,
            'constraint': constraint1
        }

        dfpd = DFPenaltyDecomposition(fun, x_0 = np.array([np.zeros(fun.number_of_x)]).transpose(), gamma=1.1, max_iterations=500000000000000, l0_constraint=constraint1, tau_zero=1, name=name, save=salva)
        print("[DP penalty decomposition on " + name + "] starting at " + time.asctime())
        start = time.time()
        dfpd.startWithRandomizedStep()
        end = time.time()
        elapsed = end-start

        res = list(np.array(dfpd.resultPoint.transpose())[0] )
        print("------> " + str(res))
        minp = list(  np.array(dfpd.minPoint.transpose())[0] )
        print(minp)
        print("Min value, B restart: " + str(dfpd.minVal))
        print("Res value " + str(dfpd.resultVal))

        res_temp['dfpd_randomized_step'] = {'elapsed-time': elapsed, 'return-point': res, 'min-point': minp, 'return-val': dfpd.resultVal, 'min-val': dfpd.minVal,
            'constraint': constraint1
        }

        X, Y = data.get_dataset()

        #print(X)
        #print(Y)
        print("[Misto interi on " + name + "] starting at " + time.asctime())
        start = time.time()
        ms = MistoInteri(X, Y, constraint1)
        end = time.time()
        elapsed = end-start


        print(np.array(ms.result[0]))
        print(ms.result[1])

        res_temp['misto-interi'] = {'elapsed-time': elapsed, 'return-point': ms.result[0], 'return-val': ms.result[1]/2,  'constraint': constraint1}

        #print(results)


        
        #resJson = json.dumps(results)
        #print(resJson)
        #Blocco successivo per salvare i risultati su file
        if True:
            results[name] = res_temp
            temp = {}
            with open(currentFileName) as infile:
                temp = json.load(infile)

            temp[name] = res_temp

            with open(currentFileName, 'w') as outfile:
                json.dump(temp, outfile, indent=4)

    
def runTests(data):
    X, Y = data.get_dataset()
    Y = np.array([Y])
    Y = Y.transpose()

    fun = RegressioneLineare(X, Y)
    constraint1 = math.floor(fun.number_of_x / divisore_constraint)


    #pendec = PenaltyDecomposition(fun, x_0 = np.array([np.zeros(fun.number_of_x)]).transpose(), gamma=1.1, max_iterations=500000000000000, l0_constraint=constraint1, tau_zero=1)
    print("[Penalty decomposition " + "] starting at " + time.asctime())
    start = time.time()
    #pendec.start()
    end = time.time()
    elapsed = end-start

    if True:
        inexact = InexactPenaltyDecomposition(fun, x_0 = np.array([np.zeros(fun.number_of_x)]).transpose(), gamma=1.1, max_iterations=500000000000000, l0_constraint=constraint1, tau_zero=1)
        print("[Inexact penalty decomposition on "  + "] starting at " + time.asctime())
        start = time.time()
        #inexact.start()
        end = time.time()
        elapsed = end-start

        dfpd = DFPenaltyDecomposition(fun, x_0 = np.array([np.zeros(fun.number_of_x)]).transpose(), gamma=1.1, max_iterations=500000000000000, l0_constraint=constraint1, tau_zero=1)
        print("[DP penalty decomposition on " + "] starting at " + time.asctime())
        start = time.time()
        dfpd.start()
        end = time.time()
        elapsed = end-start

    #print("PD: " + str(pendec.resultVal))
    #print("Inexact PD: " + inexact.resultVal)
    #print("DFPD: " + dfpd.resultVal)


if __name__ == "__main__": 
    main()


