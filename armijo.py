import numpy as np
import copy
from abstract_funzione import *

class Armijo:
    # di seguito sono variabili statiche, uno può modificarle a piacimento (mi raccomando, in caso sintassi è "Armijo.delta_k" etc.)
    # TODO chiedere quali potrebbero essere dei buoni valori di default

    delta_k = 1 #chiaro ha usato delta_k = 1 a prescindere
    #gamma = 0.5
    gamma = 0.00001 #per vedere se funziona come dovrebbe (confronto con chiaro) (prima era 1e-6)
    delta = 0.5
    alfa = 1


    @classmethod
    def armijoOnQTau(cls, fun, tau=None, x_in=None, y_in=None): #argomento cls è come il self per i metodi di 
        if(x_in is None or y_in is None or tau is None):
            print("[ARMIJO] controlla gli argomenti")

        x = copy.deepcopy(x_in)
        y = copy.deepcopy(y_in)

        alfa = cls.delta_k
        #La direzione dell'antigradiente è in discesa
        direzione = -1 * fun.getQTauXGradient(tau, x, y)

        #print("Direzione: ")
        #print(direzione)
        #print("x + alfa dot direzione: ")
        #print(x + np.dot(alfa, direzione))

        #print("Funzione valutata sul valore sopra:")
        #print(fun.getQTauValue(tau, x + alfa*direzione, y))
        
        
        
        j = 0
        #print(fun.evaluate_function( x + alfa * direzione))
        while alfa > 1e-6 and fun.getQTauValue(tau, x + np.dot(alfa, direzione), y) > fun.getQTauValue(tau, x, y) + cls.gamma*alfa*(np.dot(fun.getQTauXGradient(tau, x, y).transpose(), direzione)):
            
            alfa = cls.delta*alfa
            j+=1 #anche se in realtà non serve funzionalmente.. può però dire in quante iterazioni è terminato l'algoritmo
        
        #print("Armijo: j = " + str(j) + " alfa = " + str(alfa))
        return alfa
        