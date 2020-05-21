from regressione_lineare import *
import numpy as np
from penalty_decomposition import *
from armijo import *
from inexact_penalty_decomposition import * 
from DF_line_search import * 
from Dataset import * 
from DF_penalty_decomposition import * 

def main():

    fun = RegressioneLineare(np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]), np.array([[1],[2],[3],[4],[5]]))

    #DFLineSearch.provaLineSearch(None, alfa_zero = 1)
    #dfpd = DFPenaltyDecomposition(fun, x_0=np.array([[1],[2],[3]]), gamma=1.1, max_iterations=20, l0_constraint=2, tau_zero=2)
    #dfpd.start()
    
    #print(fun.getValueInX(np.array([[1],[2],[3]])))
    #print(fun.getValueOfGradientInX(np.array([[1],[2],[3]])))
    
    #print(fun.getQTauXGradientNorm(5, np.array([[1],[2],[3]]), np.array([[1],[2],[3]])))
    #print(fun.getQTauXGradient(5, np.array([[1],[2],[3]]), np.array([[1],[2],[3]])))
    #print(fun.getFeasibleYQTauArgminGivenX(5, np.array([[1],[2],[3]]), 2))

    #runOnServoDataset()
    #runOnHousing()

    #runOnForestFires() #inexact slower, DF only one iteration -> 275.89008406564056 (Exact --> 262)
    runOnBreastCancer()


    #runOnSmallLinearRegression()
    
    

    #print(fun.getQTauOttimoGivenY(3, np.array([[1],[2],[3]]), np.matrix([1,2,3])))
    
def runOnForestFires():
    data = Dataset(name="forest-fires", directory="./datasets/")
    run(data)


def runOnServoDataset():
    data = Dataset(name="servo", directory="./datasets/")
    run(data)
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

        #dfpd = DFPenaltyDecomposition(fun, x_0=np.array([X[0]]).transpose(), gamma=1, max_iterations=1, l0_constraint=15, tau_zero=2)
        x0 = np.array([[ 0.30989428],
            [ 0.25898083],
            [ 0.        ],
            [-0.69131161],
            [-0.15418535],
            [ 0.44026589],
            [ 0.        ],
            [-0.20055506],
            [-0.28838699],
            [-0.2844907 ],
            [ 1.64855397],
            [-0.21032222],
            [-0.9012133 ],
            [-0.72221473],
            [-0.47445776],
            [-0.26085684],
            [ 0.        ],
            [ 0.32707444],
            [ 0.        ]])

        #dfpd = DFPenaltyDecomposition(fun, x_0 = np.array([X[0]]).transpose(), gamma=1.1, max_iterations=3, l0_constraint=15, tau_zero=1)
        dfpd = DFPenaltyDecomposition(fun, x_0 = np.array([np.ones(fun.number_of_x)]).transpose(), gamma=1.1, max_iterations=3, l0_constraint=15, tau_zero=1)
        #dfpd = DFPenaltyDecomposition(fun, x_0 = x0, gamma=1.1, max_iterations=1, l0_constraint=15, tau_zero=1)
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


def runOnAutomobile():
    data = Dataset(name="automobile", directory="./datasets/")
    run(data)

def runOnHousing():
    data = Dataset(name="housing", directory="./datasets/")
    run(data)

def runOnBreastCancer():
    data = Dataset(name="breast-cancer", directory="./datasets/")
    run(data)

def run(data):

    X, Y = data.get_dataset()
    Y = np.array([Y])
    Y = Y.transpose()
    print("Shape X " + str(X.shape))
    print("Shape Y " + str(Y.shape))

    
    fun = RegressioneLineare(X, Y)
    pendec = PenaltyDecomposition(fun, x_0= np.array([X[0]]).transpose(), gamma=1.1, max_iterations=500000000, l0_constraint=15, tau_zero=1)
    pendec.start()

    inexact = InexactPenaltyDecomposition(fun, x_0=np.array([X[0]]).transpose(), gamma=1.1, max_iterations=5000000000, l0_constraint=15, tau_zero=1)
    inexact.start()

    dfpd = DFPenaltyDecomposition(fun, x_0 = np.array([np.ones(fun.number_of_x)]).transpose(), gamma=2, max_iterations=5000000000, l0_constraint=15, tau_zero=1)
    dfpd.start()
    print("Shape X " + str(X.shape))
    print("Shape Y " + str(Y.shape))

    print("PD --> " + str(pendec.resultVal))
    print("INEXACT PD --> " + str(inexact.resultVal))
    print("DF PD --> " + str(dfpd.resultVal))

if __name__ == "__main__": 
    main()


    #CHANGES: 