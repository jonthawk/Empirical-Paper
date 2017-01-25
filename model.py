"""
This file contains a "Moments" class.
Its purpose is to compute the moments G_j(\theta, S^n, P_ns)
"""

"""TODO: 
   - Do we have cost params?
   - What to do for the cost-side? Help help help...
     Maybe we just run it without IVs, and check to see if we can recover. Time to move on...

   - Brand-specific dummies, if possible (see Nevo 2000)
     + If we do brand specific dummies, we need minimum-distance procedure to identify
       all params. Chamberlain 1982... Maybe not...
  


   - Params is assumed to be length K+1, but if we have non-random characteristics,
     this will result in indeterminacy... The variance params for non-rand chars
     will not affect the moment condition, and this will cause problems. 
     + SO CAUTION: While model can be expanded to have non-random chars, it does 
       NOT currently have this capacity...

"""


import numpy as np
import statsmodels.api as sm
import statsmodels.sandbox.regression.gmm as gmm


class Moments:
    """A market is initialized with:
    1) params: Structural Parameters to be estimated 
       - Variance of taste params, inc. price sensitivity
       - Means of these params & cost params are not included
         They are implied by the params which enter non-linearly,
         and we recover them by IV regression.
       - Note: params has length (K+1), that is, one variance param 
         for each product characteristic and one for price. 

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
    7) regions: Describes which mkts are in each region
    8) non_rand_chars: Contains indices of product characteristics
         over which consumers have non_random tastes (to reduce params)
    9) P0: Label giving instructions on how to generate a
         distribution of consumers given demand_params
    10) NS: The number of draws from P0, ("Consumers in the market") 
         Note: It seems like this ought to be the number of 
           consumers sampled times S, the # of simulations. 
           However, I'm not sure it matters... May revisit...
    ++++++++
    - Since prices are so small relative to income, we assume quasi-linear utility
      
    - Need to resolve how to think about OO. OO has no price, share defined implicitly
      I think we prefer to add OO to the data, with 0 price and 0 for all characteristics
      In fact, its characteristics are irrelevant because in cond_share function, mu[0]=0 
      and delta[0] is also normalized to 0 in the end. The only remaining thing is to 
      remember to trim OO when doing various manipulations (e.g. calculating markups)


    """

    def __init__(self, params,                  
                 market_shares, prices, 
                 product_chars, cost_chars, 
                 ownership, regions,
                 non_rand_chars=[], P0='normals', NS=500):

        #Number of time periods
        self.T = len(market_shares)
        #Number of markets (cities)
        self.M = len(market_shares[0])
        #Number of products (not counting outside opt.)
        self.J = len(market_shares[0][0])-1
        #Number of firms
        self.F = len(ownership)
        #Number of demand-relevant characteristics
        self.K = len(product_chars[0])
        #Number of cost-relevant characteristics
<<<<<<< HEAD
        self.C = C

        #Now, we fold Data into data[t][mkt][j]
        #Note, for no good reason, Prices, Shares have 00, but they are irrelevant
        prices = np.zeros((self.T,self.M,self.J+1))
        shares = np.zeros((self.T,self.M,self.J+1))
        
        for obs in self.Data:
            t   = int(obs[0])
            mkt = int(obs[1])
            j   = int(obs[2])+1
            prices[t][mkt][j] = obs[3]
            shares[t][mkt][j] = obs[4]

        for t in range(self.T):
            for mkt in range(self.M):
                shares[t][mkt][0] = 1-np.sum(shares[t][mkt])

        prod_chars = [self.Data[j][6:6+self.K] for j in range(self.J)]
        cost_chars = [self.Data[j][6+self.K:6+self.K+self.C] 
                      for j in range(self.J)]

        #Coefficients on K product chars and price
        self.params     = [0.4, 3.1, 2.6, 1.9, 1.4, 1]
        self.mkt_shares = shares
=======
        self.C = len(cost_chars[0])

        #Coefficients on K product chars and price
        self.params     = params
        self.mkt_shares = market_shares
>>>>>>> parent of a6b84c1... Avoided duplication of delta computation
        self.prices     = prices
        self.prod_chars = product_chars
        self.cost_chars = cost_chars
        self.ownership  = ownership
        self.regions    = regions
        self.non_rand_chars = non_rand_chars
        self.NS = NS

        #We draw the nu's for T*NS consumers
        #P0[consumer] = [y_i, nu^1_i, ... nu^K_i]
<<<<<<< HEAD
        self.normals = self.draw_normals()
        self.P0      = self.compute_P0(self.params)
        """        
        self.mu = [ [ [np.concatenate([0], [-self.P0[t][mkt][i][0]*self.prices[t][mkt][j] +
                                             np.dot(self.prod_chars[j], self.P0[t][mkt][i][1:]) 
                                             for j in range(self.J)])
                       for i in range(self.NS)]
                      for mkt in range(self.M)]
                    for t in range(self.T)]
                    """
=======
        if P0 == "normals":
            self.P0 = self.draw_normals()
        elif P0 == "full":
            #y_i also varies, according to some emp. dist. 
            self.P0 = self.draw_full()
>>>>>>> parent of a6b84c1... Avoided duplication of delta computation
        
        #We compute the IVs
        self.Z = self.make_IVs()


    def draw_normals(self):
        """
        Under this specification:
        The nu_i are a vector of K+1 independent N(0,sigma_k) RVs
        P0[Time][market][draw][char]

        Note: If we want non-rand chars, this function will need to be updated
        """


        P0 = np.array([np.random.normal(scale=self.params[k], size=(self.M, self.NS, self.T))
              for k in range(self.K+1)])
        P0 = np.swapaxes(P0, 0, 3)

        return P0

<<<<<<< HEAD
    def compute_P0(self, params):
        """Takes parameters, produces new P0 by adjusting variances"""
        """NOTE: This means params are variances, not std"""
=======
    def draw_full(self):
        """
        Under this specification,
        We might, for example, have covariance between taste shocks.
        This would, of course, entail further modifications to, e.g. params. 
        """

        return None
>>>>>>> parent of a6b84c1... Avoided duplication of delta computation

        return [ [ [ [self.normals[t][mkt][i][k]*params[k] for k in range(self.K+1)]
                     for i in range(self.NS)]
                   for mkt in range(self.M)]
                 for t in range(self.T)]
        
    def make_z(self, t, mkt, j):
        """Takes a time, a market, and a product j,
           Returns vector of all prices of j in the same region as mkt, except mkt
        """

        #Find region R that market is in
        for region in self.regions:
            if mkt in region:
                R = region
                break
        for m in R:
            avg = 0
            if m == mkt:
                continue
            else:
                avg += self.prices[t][m][j+1]
        z = [avg/(len(R)-1)]
        z.extend(self.prod_chars[j])
        z.extend(self.cost_chars[j])
        
        return z

    def make_IVs(self):
        """Produces array of exogenous instruments"""
        Z = [[[self.make_z(t, mkt, j) for j in range(self.J)]
              for mkt in range(self.M)]
             for t in range(self.T)]

        return sm.add_constant(Z)


    def cond_choice_prob(self, delta, nu, t, mkt, i):
        """
        This function takes the J-vector of deltas and a K-vector nu_i
        It returns the conditional choice probs for product j for 
        consumer i
        """
        
        """Note: We have quasi-linear utility (in money) - no wealth effects"""

        cond_probs = np.zeros(self.J+1)

        mu = [-nu[0]*self.prices[t][mkt][j] +
               np.dot(self.prod_chars[j], nu[1:])
               for j in range(self.J)]

        D = [(delta[j] + mu[j]) for j in range(self.J+1)]
        expD = np.exp(D)
        denom = np.sum(expD)
        
        for j in range(self.J+1):
            cond_probs[j] = expD[j]/denom
                        
        return cond_probs

    def simulate_shares(self, delta, t, mkt):
        """Takes delta, 
        Returns market shares implied by the model
        """

        totals = np.zeros(self.J+1)

        for i in range(self.NS):
            totals = np.add(totals, self.cond_choice_prob(delta, self.P0[t][mkt][i], t, mkt, i))
        shares = totals/self.NS
        
        return shares

    def delta_iterate(self, delta, t, mkt):
        """Takes delta
        Returns new delta (contraction mapping)
        """
        delta1 = delta + np.log(self.mkt_shares[t][mkt]) - np.log(self.simulate_shares(delta, t, mkt))
        return delta1
                                                          
    def find_delta(self, t, mkt, tol=1e-6, max_iter=500):
        """Iterates contraction mapping until we find delta for market mkt"""
        
        #Initial guess suggested by Nevo
        delta0 = np.log(self.mkt_shares[t][mkt])-np.log(self.mkt_shares[t][mkt][0])
        delta1 = np.ones(self.J+1)

        iter = 0
        dist = 1

        while dist > tol and iter < max_iter:
            iter += 1

            delta1 = self.delta_iterate(delta0, t, mkt)
            dist = np.linalg.norm(delta0 - delta1)
            delta0 = delta1
            
            if iter % 20 == 0:
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

        for t in range(self.T):
            for mkt in range(self.M):
                delta = self.find_delta(t, mkt)[1:]
                Y.extend(delta) #more efficient than appending below           
                for j in range(self.J):
                    x = np.append(self.prod_chars[j], self.prices[t][mkt][j+1])
                    X.append(x)

        X = sm.add_constant(X)
        reg = gmm.IV2SLS(Y,X,self.Z).fit()

        beta = reg.params
        xi   = reg.resid
        
        return beta, xi
        
    """We've finished steps 1) and 2) of the evaluation of G, 
       Next, we have the functions which compute the cost-side unobservables"""

    def compute_cond_derivs(self, delta, nu, t, mkt):
        """This function takes nu and market index,
           Returns the matrix of cond. share/price derivatives
           This function is then numerically integrated (over nu)
           to produce the matrix of s/p derivs
           """

        """If we didn't have quasi-linear utility, Dmu would be more complex, leave infra. in place"""

        #F_j = cond. choice prob for product j
        F     = self.cond_choice_prob(delta, nu, t, mkt)
        #Dmu is the vector of derivatives of mu_ij w.r.t p_j
        #Note, this doesn't really make sense for the OO, but the derivs. for the OO
        # are irrelevant anyway
        Dmu   = [nu[0] for j in range(self.J+1)]
    
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

    def simulate_derivs(self, delta, t, mkt):
        """Takes 'true' delta and market,
           Returns matrix of share/price derivatives
        """
                    
        totals = np.zeros((self.J+1, self.J+1))
        for i in range(self.NS):
            totals = np.add(totals, self.compute_cond_derivs(delta, self.P0[t][mkt][i], t, mkt))
        derivs = totals/self.NS
        
        return derivs
    
    def make_Delta(self, delta, t, mkt):
        """Takes a market
           Returns the Delta matrix, where 
           Delta_jr = -ds_r/dp_j if r, j are produced by the same firm
                    = 0 otherwise
           NB: Delta is JxJ matrix. We omit the OO since it does not enter
               into the FOCs of any of the firms. We eventually need to invert
               Delta, so we cannot have this column of zeros
        """
        derivs = self.simulate_derivs(delta, t, mkt)
        Delta = np.zeros((self.J, self.J))
        for firm in self.ownership:
            for j in firm: #Iterate through all products owned by firm
                for r in firm:
                    #j-1 'trims' the first column and row
                    Delta[j-1][r-1] = -derivs[j][r]
        return Delta

    def find_markups(self, t, mkt):
        """Takes a market,
           Returns the J-vector of markups
        """
        
        delta  = self.find_delta(t, mkt)
        Delta  = self.make_Delta(delta, t, mkt)
        #We trim the share of the OO
        shares = self.simulate_shares(delta, t, mkt)[1:]
        invD = np.linalg.inv(Delta)
        b = np.dot(invD, shares)
        return b

    def find_cost_unobs(self):
        """This function computes cost parameters and unobservables
           using Hausman instruments?"""

        """For the sake of time, there will be no cost-shifters.
           That is, omega = ln(p - b(p,x,xi;theta)). Something to ask about.
           for now, we just return cost unobservables.
        """
        
        B = []
        P = []
        X = []

        for t in range(self.T):
            for mkt in range(self.M):
                B.extend(self.find_markups(t, mkt))
                P.extend(self.prices[t][mkt][1:])
                for j in range(self.J):
                    X.append(self.cost_chars[j])
        Y = np.log(np.array(P) - np.array(B))
        X = sm.add_constant(X)

        reg = sm.OLS(Y,X).fit()
        gamma = reg.params
        om   = reg.resid

        return gamma, om

    def find_Gj(self, params):
        """Takes parameters,
           Returns G_J to be minimized
        """

#        print(self.Z)
        Z = np.concatenate((self.Z, self.Z))
        Phi0 = np.dot(np.transpose(Z), Z)
#        print(Phi0)

        Phi0inv = np.matrix(np.linalg.inv(Phi0))
#        print(Phi0inv)

        beta,  xi = self.find_demand_unobs()
        gamma, om = self.find_cost_unobs()

        W  = np.matrix(np.concatenate((xi, om)))
        Z  = np.matrix(Z)
        #Transposes are nonstandard, comp. to Nevo, but correct. 
        #linalg.norm converts a 1x1 matrix into scalar, kludge
        return np.linalg.norm(W*Z*Phi0inv*Z.T*W.T)
        

        
np.random.seed(0)

num_time = 2
num_mkts = 4
num_prod = 5 #Number of products, including OO                               
                          
params = [.5, 1, 1, 1, 1, 1]    
prices = 10*np.random.rand(num_time, num_mkts, num_prod)

for t in range(num_time):
    for m in range(num_mkts):
        prices[t][m][0] = 0 #OO has price normalized to 0
#print(prices)

shares = np.random.rand(num_time, num_mkts, num_prod)

for t in range(num_time):
    shares[t] = shares[t]/shares[t].sum(axis=1)[:,None]

product_chars = np.random.rand(num_prod-1,5)
cost_chars = np.random.rand(num_prod-1, 2)

#In ownership, 0 index should not exist. 0 is to OO, and is owned by no firm
ownership = [[1,2],[3], [4]]
regions   = [[0,1], [2,3]]

foo = Moments(params, shares, prices, product_chars, cost_chars,
              ownership, regions, NS=200)

print(foo.find_Gj(params))    
