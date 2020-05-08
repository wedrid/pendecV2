from abstract_funzione import *

class RegressioneLineare(Funzione):

    def __init__(self, A, b):
        # A is a numpy matrix
        print(A.shape)
        print(b.shape)
        if len(A.shape) > 2:
            print("A non può avere più di due dimensioni")
            return
        
        if b.shape[1] > 1: 
            print("Problema malposto")
            return

        if A.shape[0] != b.shape[0]:
            print("Problema malposto")
            return
        
        self.A = A
        self.b = b
        self.n = A.shape[1]
        self.m = A.shape[0]

        print("A = \n" + str(A))
        print("b = \n" + str(b))
        print("m x n = " + str(self.m) + "x" +str(self.n))
        self.number_of_x = self.n


    #utilizzo questa classe come se fosse un'interfaccia
    def getValueInX(self, x):
        if x.shape != (self.n, 1):
            print("[getValueInX]: DIMENSIONE x ERRATA " + str(x.shape) + " EXPECTED: " + str((self.n, 1)))
            return None
        firstMember = 0.5 * np.dot(np.dot(np.dot(x.transpose(), self.A.transpose()), self.A), x)
        secondMember =  -1 * np.dot(np.dot(x.transpose(), self.A.transpose()), self.b)
        thirdMember = 0.5 * np.dot(self.b.transpose(), self.b)

        return (firstMember + secondMember + thirdMember)
    
    def getValueOfGradientInX(self, x):
        if x.shape != (self.n, 1):
            print("[getValueOfGradientInX]: DIMENSIONE x ERRATA - " + str(x.shape) + " EXPECTED: " + str((self.n, 1)))
            return None
        firstMember = np.dot(np.dot(self.A.transpose(), self.A), x)
        secondMember = -1 * np.dot(self.A.transpose(), self.b)
        return (firstMember + secondMember)


    def getQTauXGradient(self, tau, x, y):
        fGradient = self.getValueOfGradientInX(x)

        #il gradiente della "penalizzazione" è grad(||x-y||^2) = 2(x-y) (valore assoluto non c'è più quindi? TODO )
        penalGradient = tau * (x-y) #qui c'è solo tau perchè (tau/2) * 2
        qTauXGradient = (fGradient + penalGradient)
        print("Gradiente:\n " + str(qTauXGradient))
        return qTauXGradient
        

    def getQTauXGradientNorm(self, tau, x, y):
        norma = np.linalg.norm(self.getQTauXGradient(tau, x, y))
        print("Norma = " + str(norma))
        return norma
