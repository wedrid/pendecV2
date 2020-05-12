from regressione_lineare import *
import numpy as np
from penalty_decomposition import *
from armijo import *
from inexact_penalty_decomposition import * 

def main():
    fun = RegressioneLineare(np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3]]), np.array([[1],[2],[3],[4]]))
    print(fun.getValueInX(np.array([[1],[2],[3]])))
    print(fun.getValueOfGradientInX(np.array([[1],[2],[3]])))
    #print(fun.getQTauXGradientNorm(5, np.array([[1],[2],[3]]), np.array([[1],[2],[3]])))
    #print(fun.getQTauXGradient(5, np.array([[1],[2],[3]]), np.array([[1],[2],[3]])))
    #print(fun.getFeasibleYQTauArgminGivenX(5, np.array([[1],[2],[3]]), 2))

    Armijo.armijoOnQTau(fun, tau=2, x_in=np.array([[1],[2],[3]]), y_in=np.array([[1],[1],[1]]))

    pendec = PenaltyDecomposition(fun, x_0=np.array([[1],[2],[3]]), gamma=1.2, max_iterations=5, l0_constraint=2, tau_zero=5)
    pendec.start()

    inexact = InexactPenaltyDecomposition(fun, x_0=np.array([[1],[2],[3]]), gamma=1.2, max_iterations=5, l0_constraint=2, tau_zero=5)
    inexact.start()
    #print(fun.getQTauOttimoGivenY(3, np.array([[1],[2],[3]]), np.matrix([1,2,3])))

if __name__ == "__main__": 
    main()