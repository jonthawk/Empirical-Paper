"""
This file contains a "Moments" class.
Its purpose is to compute the moments G_j(\theta, S^n, P_ns)
"""

import numpy as np

class Moments:
    """A market is initialized with:
    1) params: Structural Parameters to be  
    2) market_shares: Contains observed market shares
        market_shares[market][product]
    3) prices: Contains observed prices
         prices[market][product]
    4) product_chars: Contains product characteristics affecting demand
         product_chars[product] = [char1, char2, ... ]
    5) cost_chars: Contains product characteristics affecting cost
         cost_chars[product] = [char1, ...]
         May overlap with product chars
    6) ownership: Describes which products are owned by each firm
         ownership[firm] = list of product indices owned by firm
    7) non_rand_chars: Contains indices of product characteristics
         over which consumers have non_random tastes (to reduce params)
    7) P0: Label giving instructions on how to generate a
         distribution of consumers given demand_params
    8) NS: The number of draws from P0, ("Consumers in the market") 
         Note: It seems like this ought to be the number of 
           consumers sampled times S, the # of simulations. 
           However, I'm not sure it matters... May revisit...
    """

    def __init__(self, params,                  
                 market_shares, prices, 
                 product_chars, cost_chars, 
                 ownership, non_rand_chars,
                 P0='standard', NS=500):

        #Number of markets
        self.T = len(market_shares)
        #Number of products (not counting outside opt.)
        self.J = len(market_shares[0])-1
        #Number of firms
        self.F = len(ownership)
        #Number of demand-relevant characteristics
        self.K = len(product_chars[0])
        #Number of cost-relevant characteristics
        self.C = len(cost_chars[0])

        #Coefficients on K product chars and price
        self.d_params   = params[0:self.K+1]
        #Variances of Random Coefficients
        if P0 != 'standard': #Here, there are also K variances to estimate
            self.v_params = params[(self.K+1):(2*self.K + 1)]
            #Coefficients on cost_chars
            self.c_params = params[2*self.K+1:]
        else:
            self.c_params = params[self.K+1:]

        self.mkt_shares = market_shares
        self.prices     = prices
        self.prod_chars = product_chars
        self.ownership  = ownership
        self.non_rand_chars = non_rand_chars
        self.NS = NS

        #We draw the nu's for T*NS consumers
        #P0[consumer] = [y_i, nu^1_i, ... nu^K_i]
        if P0 == "standard":
            self.P0 = self.draw_standard()
        elif P0 == "normals":
            self.P0 = self.draw_normals()
        elif P0 == "full":
            #y_i also varies, according to some emp. dist. 
            self.P0 = self.draw_full()

    def draw_standard(self):
        """
        Under this specification, len(d_params)=self.K
        We hold y_i fixed at 50,233, i.e. median income in 2007
        The nu_i is an array of self.K standard normal RVs
        """

        P0 = np.random.normal(size=(self.T, self.NS, self.K+1))
        
        #We set y_i to 50,233, and any characteristics
        #without random coefficients get nu_k = 0
        for t in range(self.T):
            for i in range(self.NS):
                P0[t][i][0] = 50233
                for char in self.non_rand_chars:
                    P0[t][i][char] = 0

        return P0
                
    def draw_normals(self):
        """
        Under this specification,
        We again hold y_i fixed at 50,233
        The nu_i is an array of self.K N(0,sigma_k)
        """
        return None

    def draw_full(self):
        """
        Under this specification,
        We draw y_i randomly from an empirical distribution
        The nu_i array is as in draw_normals
        """

        return None

    def cond_choice_prob(self, delta, nu, mkt):
        """
        This function takes the J-vector of deltas and a K-vector nu_i
        It returns the conditional choice probs for product j for 
        consumer i
        """

        cond_probs = np.zeros(self.J+1)
        
        mu = [ (self.d_params[0]*np.log(nu[0] - self.prices[mkt][j]) +
                np.dot(self.prod_chars[j], nu[1:])) for j in range(self.J)]
        mu.insert(0,0) #Outside option is 0

        D = [(delta[j] + mu[j]) for j in range(self.J+1)]
        expD = np.exp(D)
        denom = np.sum(expD)
        
        for j in range(self.J+1):
            cond_probs[j] = expD[j]/denom
            
        return cond_probs

    def simulate_shares(self, delta, mkt):
        """Takes delta, 
        Returns market shares implied by the model
        """

        totals = np.zeros(self.J+1)
        
        for i in range(self.NS):
            totals = np.add(totals, self.cond_choice_prob(delta, self.P0[mkt][i], mkt))
        shares = totals/self.NS
        
        return shares

    def delta_iterate(self, delta, mkt):
        """Takes delta
        Returns new delta (contraction mapping)
        """

        delta1 = delta + np.log(self.mkt_shares[mkt]) - np.log(self.simulate_shares(delta, mkt))
#        print("mkt_shares: ", self.mkt_shares[mkt])
#        print("sim_shares: ", self.simulate_shares(delta, mkt))
#        print(np.log(self.mkt_shares[mkt]) - np.log(self.simulate_shares(delta, mkt)))


#        print(delta1)
#        delta1[0] = 0
        return delta1
                                                          
    def find_delta(self, mkt, tol=1e-4, max_iter=500):
        """Iterates contraction mapping until we find delta for market mkt"""

        delta0 = np.zeros(self.J+1)
        delta1 = np.ones(self.J+1)

        iter = 0
        dist = 1

        while dist > tol and iter < max_iter:
            iter += 1

            delta1 = self.delta_iterate(delta0, mkt)
            dist = np.linalg.norm(delta0 - delta1)
            delta0 = delta1
            
            if iter % 10 == 0:
                print(iter, ": ", dist)

        return delta0 

    def find_demand_unobs(self, mkt):
        """This function takes a market index,
           Returns the J+1 vector of demand unobservables,
           where xi[0] is normalized to 0"""

        xB = [ (np.dot(self.prod_chars[j], self.d_params[1:])) for j in range(self.J)]
        xB.insert(0,0) #Outside option is 0        
    
        delta = self.find_delta(mkt)
        xi = delta - xB
        return xi
        
    """We've finished steps 1) and 2) of the evaluation of G, 
       Next, we have the functions which compute the cost-side unobservables"""

    def compute_cond_derivs(self, delta, nu, mkt):
        """This function takes nu and market index,
           Returns the matrix of cond. share/price derivatives
           This function is then numerically integrated (over nu)
           to produce the matrix of s/p derivs
           """

        #F_j = cond. choice prob for product j
        F     = self.cond_choice_prob(delta, nu, mkt)
        #Dmu is the vector of derivatives of mu_ij w.r.t p_j
        Dmu   = [-self.d_params[0]/(nu[0] - self.prices[mkt][j]) for j in range(self.J+1)]
    
        #Initialize cond. derivatives matrix (J+1 products, counting OO)
        cD = np.zeros((self.J+1, self.J+1))
        
        for j in range(self.J+1):
            for q in range(j+1):
                if j == q:
                    cD[j][q] = F[j]*(1-F[j])*Dmu[j]
                else:
                    cD[j][q] = -F[j]*F[q]*Dmu[q]
                    cD[q][j] = cD[j][q] #Fill by sym.
        return cD

    def simulate_derivs(self, mkt):
        #Compute delta from the data
        delta = self.find_delta(mkt)
                    
        totals = np.zeros(self.J+1)
        for i in range(self.NS):
            totals = np.add(totals, self.compute_cond_derivs(delta, self.P0[mkt][i], mkt))
        derivs = totals/self.NS
        
        return derivs
    
        
np.random.seed(0)
                                                         
#d_params  = [  0, 0, 0, 0, 0, 0] 
d_params = [0.1, 1, 1, 1, 1, 1]    
c_params = []
v_params = []

params = d_params+v_params+c_params

prices = 10*np.random.rand(4,3) #4 markets, 2 products (+outside)
shares = np.random.rand(4,3)
shares = shares/shares.sum(axis=1)[:,None]

product_chars = np.random.rand(2,5)
cost_chars = [ [ [] for j in range(3)] for t in range(4)]

ownership = [[1], [2]]
non_rand_chars = []

foo = Moments(params, shares, prices, product_chars, cost_chars,
              ownership, non_rand_chars, NS=500)

print(foo.simulate_derivs(0))    
