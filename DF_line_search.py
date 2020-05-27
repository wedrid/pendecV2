import numpy as np

class DFLineSearch:
    gamma =  0.00001
    sigma = 2 #orig 1.1

    
    #NOTA line search sembra funzionare correttamente (verificato tramite funz. semplice x**2+y**2)
    @classmethod
    def lineSearchOnQTau(cls, fun, tau=None, d=None, alfa_zero = None, x_in=None, y_in=None): #ha y come valore, ma rimane fissato, solo che per il calcolo del valore è necessario
        alfa = alfa_zero
        if alfa == 0:
            print("Alfa zero non può essere zero")
            return
        
        if fun.getQTauValue(tau, (x_in + alfa*d), y_in) <= fun.getQTauValue(tau, x_in, y_in) - cls.gamma * ( alfa**2 ) * (( np.linalg.norm(d) )**2  ):
            beta = alfa

            while True:
                alfa = beta
                beta = cls.sigma * alfa
                if fun.getQTauValue(tau, (x_in + beta*d), y_in) > fun.getQTauValue(tau, x_in, y_in) - cls.gamma * ( beta**2 ) * (( np.linalg.norm(d) )**2  ):
                    break 
            
            return alfa
        
        #else.. perchè se entra sicuramente ritorna prima di arrivare qua
        alfa = 0.0
        
        return alfa


    @classmethod
    def provaLineSearch(cls, fun, tau=None, d=None, alfa_zero = None, x_in=None, y_in=None): #ha y come valore, ma rimane fissato, solo che per il calcolo del valore è necessario
        alfa = alfa_zero
        if alfa == 0:
            print("Alfa zero non può essere zero")
            return
        x = np.array([[10,10]])
        d = np.array([[-1,-1]])
        if DFLineSearch.f(x + alfa*d) <= DFLineSearch.f(x) - cls.gamma * ( alfa**2 ) * (( np.linalg.norm(d) )**2  ):
            beta = alfa

            while True:
                alfa = beta
                beta = cls.sigma * alfa
                print("\t\t\t\t\t\t\t\t\t\t BETA: " + str(beta))
                print(DFLineSearch.f(x + alfa*d))
                if DFLineSearch.f(x + alfa*d) > DFLineSearch.f(x) - cls.gamma * ( beta**2 ) * (( np.linalg.norm(d) )**2):
                    break 
            
            print(DFLineSearch.f(x + alfa*d))
            print("Returned: " + str(alfa))
            return alfa
        
        #else.. perchè se entra sicuramente ritorna prima di arrivare qua
        alfa = 0.0
        print("direzione in salita")
        return alfa

    @classmethod
    def f(self, x):
        return x[0][0]**2 + x[0][1]**2