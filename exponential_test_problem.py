from abstract_funzione import *

class ExponentialTestProblem(Funzione):

    def __init__(self):
        #m, N sono le dimensioni dei vari vettori e matrici
        #TODO generazione delle matrici (discorso della distribuzione uniforme in -1,1, etc.)
        #i seguenti valori dovranno essere poi passati come argomenti
        N = 20
        m = 8
        p = 2
        s = 2


        #1 generazizone di A tramite distribuzione di probabilità uniforme in -1, 1 e singular value decomposition A in R^m*N
        self.A = np.zeros((N,N))
        for i in range(0,N):
            for j in range(0,N):
                temp = np.random.uniform(low=-0.999999999, high=1)
                self.A[i,j] = temp
        
        print(self.A)
        #2 generazione di B tramite procedimento descritto 
        #3 generazione di b tramite procedimento a pagina 13

    #returns value in X
    def getValueInX(self, x):
        # TODO implementazione della funzione e ritorno del valore
        pass

    #returns value of gradient in X
    def getValueOfGradientInX(self, x):
        #da fare numericamente
        pass 

    def getQTauXGradient(self, tau, x, y):
        #questa funzione può rimanere invariata. E' da implementare la dipendenza però 
        fGradient = self.getValueOfGradientInX(x)

        #il gradiente della "penalizzazione" è grad(||x-y||^2) = 2(x-y) (valore assoluto non c'è più quindi? TODO )
        try:
            penalGradient = tau * (x-y) #qui c'è solo tau perchè (tau/2) * 2
        except:
            print("ERRORE")
            print("x --> " + str(x))
            print("y --> " + str(y))
        qTauXGradient = (fGradient + penalGradient)
        #print("Gradiente:\n " + str(qTauXGradient))
        return qTauXGradient
        

    def getQTauXGradientNorm(self, tau, x, y):
        norma = np.linalg.norm(self.getQTauXGradient(tau, x, y))
        print("Norma = " + str(norma))
        return norma

    def getFeasibleYQTauArgminGivenX(self, tau, x, constraint):
        #TODO minimizzazione tramite qualche metodo di ottimizzazione
        pass

    def getQTauOttimoGivenY(self, tau, y, x0):
        #secondo blocco di ottimizzazione, anche qua da minimizzare in qualche modo
        pass