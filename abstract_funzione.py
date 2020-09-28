import numpy as np
from scipy.optimize import minimize
import copy


class Funzione:
    #utilizzo questa classe come se fosse un'interfaccia
    def getValueInX(self, x):
        return "This is abstract"
    
    def getQTauXArgminGivenY(self, tau, y):
        return "This is abstract (non dovrebbe essere usata da nessuna parte, comunque)"
    
    # TODO this seems to be [BROKEN]
    def getFeasibleYQTauArgminGivenX(self, tau, x_temp, constraint):
        #self.xTemp = x
        #self.tauTemp = tau
        #ottimo = minimize(self.getValueOfPenalizationTermHelper, x, method='BFGS')
        #print(ottimo)
        #x = ottimo.x
        x = copy.deepcopy(x_temp)
        self._makeFeasible(x, constraint)
        #print("[getFeasibleYQTauArgminGivenX] -> " + str(x))
        x = np.array(x).T
        return x

    
    def getValueOfPenalizationTermHelper(self, y):
        #print(self.xTemp)
        #print(y)
        return np.linalg.norm(self.xTemp-y)**2

    def getValueOfPenalizationTerm(self, tau, x, y): #TODO check
        return 0.5 * tau * (np.linalg.norm(x-y))**2

    def getQTauXGradientNorm(self, tau, x, y):
        return "This is abstract"

    def getQTauXGradient(self, tau, x, y):
        return "This is abstract"

    def getQTauOttimoGivenY(self, tau, y, x0): #x0 è da dove inizia la ricerca
        #general implementation
        self.yTemp = y
        self.tauTemp = tau
        #print("y = " + str(self.yTemp) + " x0 = " + str(x0))
        ottimo = minimize(self.qTauValueHelper, x0, method='BFGS')
        #ottimo = minimize(self.qTauValueHelper, x0, method='Nelder-Mead')
        x = np.array([ottimo.x]).T
        return x

    def getQTauValue(self, tau, x, y):
        firstMember = self.getValueInX(x)
        #print("firstMember = " + str(firstMember))
        secondMember = 0.5 * tau * (np.linalg.norm(x-y))**2
        #print("secondMember = " + str(secondMember))

        return (firstMember + secondMember) #dovrebbero essere scalari, controllare TODO

    #i seguenti due metodi sono helpers per la funzione optimize di scypi
    def qTauValueHelper(self, x):
        xt = np.matrix(x).transpose()
        #print(">>>>>>>>>>>>>>>> xt\n" + str(xt))
        return self.getQTauValue(self.tauTemp, xt, self.yTemp)

    def qTauGradHelper(self, x):
        xt = np.matrix(x).transpose()
        #print("[GradHelper]>>>>>>>>>>>>>>>> xt\n" + str(xt))

        ret = self.getQTauXGradient(self.tauTemp, xt, self.yTemp).transpose()
        #print("RETURNNNNNNNNNNN" + str(np.squeeze(np.asarray(ret))))
        return np.squeeze(np.asarray(ret)) #attenzione: qui stavo ritornando una matrice, ma a minimize non piace, vuole un array
    

    def _makeFeasible(self, point, constraint):
        pointCopy = copy.deepcopy(point)
        mins_list = []
        pointCopy = np.absolute(pointCopy)
        #print(pointCopy)
        #non uso il sort perchè potrebbero esserci ripetizioni e dovrei fare ulteriori controlli
        for i in range(0, (len(point) - constraint)):
            mins_list.append(pointCopy.argmin())
            pointCopy[mins_list[i]] = float('inf')
        
        for index in mins_list:
            point[index] = 0