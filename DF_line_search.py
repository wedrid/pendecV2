import numpy as np

class DFLineSearch:
    gamma = 0.5
    sigma = 1.1

    

    @classmethod
    def lineSearchOnQTau(cls, fun, tau=None, d=None, alfa_zero = None, x_in=None, y_in=None): #ha y come valore, ma rimane fissato, solo che per il calcolo del valore è necessario
        alfa = alfa_zero
        
        if fun.getQTauValue(tau, (x_in + alfa*d), y_in) <= fun.getQTauValue(tau, x_in, y_in) - cls.gamma * ( alfa**2 ) * (( np.linalg.norm(d) )**2  ):
            beta = alfa

            while True:
                alfa = beta
                beta = cls.sigma * alfa
                if fun.getQTauValue(tau, (x_in + beta*d), y_in) > fun.getQTauValue(tau, x_in, y_in) - cls.gamma * ( alfa**2 ) * (( np.linalg.norm(d) )**2  ):
                    break 
            
            return alfa
        
        #else.. perchè se entra sicuramente ritorna prima di arrivare qua
        alfa = 0
        return alfa