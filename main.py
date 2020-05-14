from regressione_lineare import *
import numpy as np
from penalty_decomposition import *
from armijo import *
from inexact_penalty_decomposition import * 
from DF_line_search import * 
from Dataset import * 

def main():

    
    
    
    #print(fun.getValueInX(np.array([[1],[2],[3]])))
    #print(fun.getValueOfGradientInX(np.array([[1],[2],[3]])))
    
    #print(fun.getQTauXGradientNorm(5, np.array([[1],[2],[3]]), np.array([[1],[2],[3]])))
    #print(fun.getQTauXGradient(5, np.array([[1],[2],[3]]), np.array([[1],[2],[3]])))
    #print(fun.getFeasibleYQTauArgminGivenX(5, np.array([[1],[2],[3]]), 2))
    fun = RegressioneLineare(np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]), np.array([[1],[2],[3],[4],[5]]))

    if(True):
        Armijo.armijoOnQTau(fun, tau=2, x_in=np.array([[1],[2],[3]]), y_in=np.array([[1],[1],[1]]))
        alfa = DFLineSearch.lineSearchOnQTau(fun, tau=2, x_in=np.array([[1],[2],[3]]), y_in=np.array([[1],[1],[1]]), d=np.array([[0],[0],[1]]), alfa_zero=1)
        print("Alfa = " +  str(alfa))
        alfa = DFLineSearch.lineSearchOnQTau(fun, tau=2, x_in=np.array([[1],[2],[3]]), y_in=np.array([[1],[1],[1]]), d=np.array([[0],[1],[0]]), alfa_zero=1)
        print("Alfa = " +  str(alfa))
        alfa = DFLineSearch.lineSearchOnQTau(fun, tau=2, x_in=np.array([[1],[2],[3]]), y_in=np.array([[1],[1],[1]]), d=np.array([[1],[0],[0]]), alfa_zero=1)
        print("Alfa = " +  str(alfa))
        alfa = DFLineSearch.lineSearchOnQTau(fun, tau=2, x_in=np.array([[1],[2],[3]]), y_in=np.array([[1],[1],[1]]), d=np.array([[0],[0],[-1]]), alfa_zero=1)
        print("Alfa = " +  str(alfa))
        alfa = DFLineSearch.lineSearchOnQTau(fun, tau=2, x_in=np.array([[1],[2],[3]]), y_in=np.array([[1],[1],[1]]), d=np.array([[0],[-1],[0]]), alfa_zero=1)
        print("Alfa = " +  str(alfa))
        alfa = DFLineSearch.lineSearchOnQTau(fun, tau=2, x_in=np.array([[1],[2],[3]]), y_in=np.array([[1],[1],[1]]), d=np.array([[-1],[0],[0]]), alfa_zero=1)
        print("Alfa = " +  str(alfa))

    #runOnServoDataset()
    #runOnSmallLinearRegression()
    

    #print(fun.getQTauOttimoGivenY(3, np.array([[1],[2],[3]]), np.matrix([1,2,3])))
    

def runOnServoDataset():
    data = Dataset(name="servo", directory="./datasets/")
    X, Y = data.get_dataset()
    Y = np.array([Y])
    Y = Y.transpose()
    print("Shape X " + str(X.shape))
    print("Shape Y " + str(Y.shape))

    fun = RegressioneLineare(X, Y)
    pendec = PenaltyDecomposition(fun, x_0= np.array([X[0]]).transpose(), gamma=1.1, max_iterations=5, l0_constraint=15, tau_zero=2)
    #pendec.start()

    inexact = InexactPenaltyDecomposition(fun, x_0=np.array([X[0]]).transpose(), gamma=1, max_iterations=2, l0_constraint=15, tau_zero=2)
    inexact.start()


def runOnSmallLinearRegression():
    fun = RegressioneLineare(np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]), np.array([[1],[2],[3],[4],[5]]))
    pendec = PenaltyDecomposition(fun, x_0=np.array([[1],[2],[3]]), gamma=1.1, max_iterations=5, l0_constraint=3, tau_zero=2)
    
    #pendec = PenaltyDecomposition(fun, x_0= X[0], gamma=1.2, max_iterations=5, l0_constraint=2, tau_zero=5)
    pendec.start()

    inexact = InexactPenaltyDecomposition(fun, x_0=np.array([[1],[2],[3]]), gamma=1.1, max_iterations=5, l0_constraint=3, tau_zero=2)
    inexact.start()




if __name__ == "__main__": 
    main()
