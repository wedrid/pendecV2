from abstract_funzione import *
from scipy.stats import ortho_group
from scipy import linalg

#TODO yet to be implemented

class QuadraticTestProblem(Funzione):

    def __init__(self):
        #TODO generazione delle matrici (discorso della distribuzione uniforme in -1,1, etc.)
        #Sono da definire le matrici: A, Hi
        #1 definisco Q matrice ortogonale 
        #n+s definisce la sparsità di x segnato
        N = 10 #definisce il numero di variabili e tutto ciò che ne è collegato
        n = 5
        s = 3
        m = 1
        self.n = n
        self.s = s
        self.m = m
        self.N = N
        if n+s > N:
            print("Dev'essere rispettato che n+s<=N !!")
            return None
        
        Q = ortho_group.rvs(dim=n+s)
        np.set_printoptions(suppress=True) #mostra i numeri approssimati se troppo piccoli
        print(Q)
        res = np.dot(Q, Q.T)
        print(res)
        Q1 = Q[0:Q.shape[1],0:n]
        Q2 = Q[0:Q.shape[1],n:n+s]
        self.Q1 = Q1
        self.Q2 = Q2
        print(Q1)
        print(Q2)

        # 2 Genero B e C come matrici di dimensioni corrette 
        # B dev'essere \in M(N x n, R)
        # 2.1 calcolo A:=(BQ1', C) --> BQ1' risulta essere di dimensione N x (n+s)
        # da cui C di dimensione N x (N-(n+s))
        B = np.zeros((N, n))
        for i in range(0, N):
            for j in range(0,n):
                temp = np.random.uniform(low=-0.999999999, high=1)
                B[i,j] = temp
        print("B uguale ")
        print(B)
        BQ1transp = np.dot(B, Q1.T)
        print(BQ1transp)

        C = np.zeros((N, N-(n+s)))
        for i in range(0,N):
            for j in range(0,N-(n+s)):
                temp = np.random.uniform(low=-0.999999999, high=1)
                C[i,j] = temp
        print(C)
        A = np.hstack((BQ1transp, C))
        print(A)

        # 3 generazione delle matrici Hi
        self.generateHi() #da chiamare iterativamente per generare le varie righe 

        
    def generateHi(self):
        #Nota Hi dev'essere in N x N
        Ti = np.zeros((self.n, self.n))
        for i in range(0, self.n):
            for j in range(0, self.n):
                Ti[i,j] = np.random.uniform(low=-0.999999999, high=1)
        print(Ti.shape)
        print(self.Q1.shape)
        temp = np.dot(self.Q1, Ti)
        print(temp.shape)
        Q1TiQ1t = np.dot(temp, self.Q1.T)
        print(Q1TiQ1t.shape)

        #genero Si e Ri delle dimensioni corrette
        Si = np.zeros((self.n+self.s, self.N-(self.n+self.s)))
        
        for i in range(0, self.n+self.s):
            for j in range(0, self.N-(self.n+self.s)):
                Si[i,j] = np.random.uniform(low=-0.999999999, high=1)
        
        Ri = np.zeros((self.N-(self.n+self.s), self.N-(self.n+self.s)))
        print(Ri)
        
        for i in range(0, self.N-(self.n+self.s)):
            for j in range(0, self.N-(self.n+self.s)):
                Ri[i,j] = np.random.uniform(low=-0.999999999, high=1)
        print(Ri)

        hirow = np.hstack((Q1TiQ1t, Si))
        print(hirow.shape)
        lorow = np.hstack((Si.T, Ri))
        Hi = np.vstack((hirow, lorow))
        print(Hi.shape)
        return Hi

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