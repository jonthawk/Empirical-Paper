"""
This file contains a "Moments" class.
Its purpose is to compute the moments G_j(\theta, S^n, P_ns)
"""

import numpy as np
import statsmodels.api as sm


class Moments:
    """A market is initialized with:
    1) params: Structural Parameters to be estimated 
       - Price sensitivity and variance of taste params 
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
    ++++++++
    - We need to consider how to think about price sensitivity.
      in BLP log specification, price/share derivative has a alpha/(y - p) term
      When y is significantly larger than p (as in the case of beer and other 
      retail products), this means that alpha must be extremely large, otherwise
      consumers just don't care about prices at all... It's not clear whether this
      is actually problematic (alpha = 1000?) but super small ds/dp implies super
      large markups - possibly implying negative MC, which is bad! 

    - Problem? - alpha can't go above about 65, otherwise mu becomes large 
      and exp(mu) -> \inf, everything explodes. 

    - Need to resolve how to think about OO. OO has no price, share defined implicitly
      I think we prefer to add OO to the data, with 0 price and 0 for all characteristics
      In fact, its characteristics are irrelevant because in cond_share function, mu[0]=0 
      and delta[0] is also normalized to 0 in the end. The only remaining thing is to 
      remember to trim OO when doing various manipulations (e.g. calculating markups)


    """

    def __init__(self, params,                  
                 market_shares, prices, 
                 product_chars, cost_chars, 
                 ownership, non_rand_chars,
                 P0='normals', NS=500):

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
        self.params     = params
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

        P0 = np.array([np.random.normal(scale=self.params[k], size=(self.NS, self.T))
              for k in range(self.K+1)])
        P0 = np.swapaxes(P0, 0, 2)
        
        for t in range(self.T):
            for i in range(self.NS):
                P0[t][i][0] = 50233
                for char in self.non_rand_chars:
                    P0[t][i][char] = 0
        
        return P0

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
        
        mu = [ (self.params[0]*np.log(nu[0] - self.prices[mkt][j]) +
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
        #Normalize d_OO = 0
        delta0 = delta0 - delta0[0]

        return delta0 

    def find_demand_unobs(self):
        """This function computes mean tastes (beta) and demand unobservables (xi),
        """

        """Useful optimization: deltas for all markets are used here and also
           in cost unobservables. Instead of computing twice, could add them as input. 
        """

        #delta_jt, will be independent var. 
        Y = []
        X = []
        for mkt in range(self.T):
            delta = self.find_delta(mkt)[1:]
            Y.extend(delta)
            X.extend(self.prod_chars) #same in all mkts - ineffiencent
        
        reg = sm.OLS(Y,X).fit()
        
        beta = reg.params
        xi   = reg.resid
        
        return beta, xi
        
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

    def simulate_derivs(self, delta, mkt):
        """Takes 'true' delta and market,
           Returns matrix of share/price derivatives
        """
                    
        totals = np.zeros(self.J+1)
        for i in range(self.NS):
            totals = np.add(totals, self.compute_cond_derivs(delta, self.P0[mkt][i], mkt))
        derivs = totals/self.NS
        
        return derivs
    
    def make_Delta(self, delta, mkt):
        """Takes a market
           Returns the Delta matrix, where 
           Delta_jr = -ds_r/dp_j if r, j are produced by the same firm
                    = 0 otherwise
           NB: Delta is JxJ matrix. We omit the OO since it does not enter
               into the FOCs of any of the firms. We eventually need to invert
               Delta, so we cannot have this column of zeros
        """
        derivs = self.simulate_derivs(delta, mkt)
        Delta = np.zeros((self.J, self.J))

        for firm in self.ownership:
            for j in firm: #Iterate through all products owned by firm
                for r in firm:
                    #j-1 'trims' the first column and row
                    Delta[j-1][r-1] = -derivs[j][r]
        return Delta

    def find_markups(self, mkt):
        """Takes a market,
           Returns the J-vector of markups

           -Potential bug: The markups are very, very large.
            Likely a problem stemming from the alpha/(y-p) term,
            where y-p is large, making price sensitivity low for 
            reasonable alpha. Possible solution is to replace this
            with -alpha*p. 
        """
        delta = self.find_delta(mkt)
        
        Delta  = self.make_Delta(delta, mkt)
        #We trim the share of the OO
        shares = self.simulate_shares(delta, mkt)[1:]

        invD = np.linalg.inv(Delta)
        b = np.dot(invD, shares)
        return b




        
np.random.seed(0)
                                                         
params = [60, 1, 1, 1, 1, 1]    
prices = 100*np.random.rand(4,3)+5 #4 markets, 2 products (+outside)
prices[0] = 0 #OO has price normalized to 0
shares = np.random.rand(4,3)
shares = shares/shares.sum(axis=1)[:,None]

product_chars = np.random.rand(2,5)
cost_chars = [ [ [] for j in range(3)] for t in range(4)]

ownership = [[1], [2]]
non_rand_chars = []

foo = Moments(params, shares, prices, product_chars, cost_chars,
              ownership, non_rand_chars, NS=1000)

foo.draw_normals()

print(foo.find_demand_unobs())    
