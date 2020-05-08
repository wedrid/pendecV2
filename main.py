from regressione_lineare import *
import numpy as np
from penalty_decomposition import *

def main():
    fun = RegressioneLineare(np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3]]), np.array([[1],[2],[3],[4]]))
    #print(fun.getValueInX(np.array([[1],[2],[3]])))
    #print(fun.getValueOfGradientInX(np.array([[1],[2],[3]])))
    #print(fun.getQTauXGradientNorm(5, np.array([[1],[2],[3]]), np.array([[1],[2],[3]])))
    #print(fun.getQTauXGradient(5, np.array([[1],[2],[3]]), np.array([[1],[2],[3]])))
    #print(fun.getFeasibleYQTauArgminGivenX(5, np.array([[1],[2],[3]]), 2))

    pendec = PenaltyDecomposition(fun, x_0=np.array([[1],[2],[3]]), gamma=20, max_iterations=3, l0_constraint=2, tau_zero=4)
    pendec.start()


    #print(fun.getQTauOttimoGivenY(3, np.array([[1],[2],[3]]), np.matrix([1,2,3])))

if __name__ == "__main__": 
    main()