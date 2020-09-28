from abstract_funzione import *
from scipy import optimize


class RegressioneLineare(Funzione):

    def __init__(self, A, b):
        # A is a numpy matrix
        print("Shape of A: " + str(A.shape))
        print("Shape of b: " + str(b.shape))
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

        return (firstMember + secondMember + thirdMember)[0][0]
    
    def getValueOfGradientInX(self, x):
        if x.shape != (self.n, 1):
            print("[getValueOfGradientInX]: DIMENSIONE x ERRATA - " + str(x.shape) + " EXPECTED: " + str((self.n, 1)))
            return None
        firstMember = np.dot(np.dot(self.A.transpose(), self.A), x)
        secondMember = -1 * np.dot(self.A.transpose(), self.b)
        gradient = firstMember + secondMember
        #grad = optimize.approx_fprime(x, self.getValueInX, 1e-6)


        return gradient


    def getQTauXGradient(self, tau, x, y):
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
        return self.chiariniAnaliticSM(x, constraint)

    def chiariniAnaliticSM(self,u,s):
        i=0
        u_temp=[]
        while i<len(u):
            if type(u[i])==int:
                if(u[i])<0:
                    u_temp.insert(i,-u[i])
                else:
                    u_temp.insert(i,u[i])
            else:
                if (u[i][0])<0:
                    u_temp.insert(i,-u[i][0])
                else:
                    u_temp.insert(i,u[i][0])
            i+=1
        u_temp.sort()
        u_temp.reverse()
        u=np.array(u)
        i=s
        while i<len(u_temp):
            j=0
            flag=0
            while j<len(u) and flag==0:
                if u_temp[i]==u[j][0] or u_temp[i]==-u[j][0]:
                    flag=1
                if flag!=1:
                    j+=1
            if flag==1:
                u[j][0]=0
            i+=1
        x=[]
        k=0
        while k<len(u):
            x.append([])
            x[k].insert(0,u[k][0])
            k+=1
        y=np.array(x)
        return y.transpose()

    def getQTauOttimoGivenY(self, tau, y, x0):
        n = self.n

        At=np.transpose(self.A)
        first=np.dot(At,self.A)+np.identity(n)*tau
        first = np.matrix(first, dtype='float')
        first= np.linalg.inv(first)
        second=np.dot(At,self.b)+tau*y
        x_star=np.dot(first,second)

        return x_star