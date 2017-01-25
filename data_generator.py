"""This file contains class market. 
   It takes an array of true parameters
   generates a dataset for verifying estimation procedure"""

import numpy as np
import csv


class Market:
    """A market is initialized with:
    1) theta: a non-linear parameter array
       contains price-sensitivity and variances of taste parameters
    2) beta: a linear parameter array
       contains the mean taste params - inc. price sense. 
    3) gamma: a linear parameter array
       contains the cost shifter params
    """

    def __init__(self, theta, beta, gamma, 
                 ownership, prod_chars, cost_chars,
                 marg_cost, NS=1000):
        self.theta = theta
        self.beta  = beta
        self.gamma = gamma

        self.NS = NS
        
        #Num products not counting OO
        self.J = len(prod_chars)
        #Num of product characteristics
        self.K = len(prod_chars[0])

        self.ownership  = ownership
        self.prod_chars = prod_chars
        self.cost_chars = cost_chars

        #Demand unobservable for each product (except OO)
        self.xi = np.random.normal(size=self.J)
        self.MC = marg_cost + np.dot(self.cost_chars, self.gamma)
        self.P0 = self.draw_normals()

    def draw_normals(self):
        """Draws NS nu_i vectors
        nu_i are K+1 independent N(0, sigma_k) RVs
        """

        P0 = np.array([np.random.normal(scale=self.theta[k], size=self.NS) 
              for k in range(self.K+1)])
        
        return np.transpose(P0)

    def cond_choice_prob(self, prices, nu):
        """Takes matrix of prices and array nu_i
        returns conditional choice probabilities
        Taste shocks are additive
        """
        
        cond_probs = np.zeros(self.J+1)

        beta_i = [beta[k+1]+nu[k+1] for k in range(self.K)]
        D = [np.dot(self.prod_chars[j], beta_i) 
             - (beta[0]+nu[0])*prices[j]
             + self.xi[j] for j in range(self.J)]

        D.insert(0,0) #OO is 0
        expD = np.exp(D)
        denom = np.sum(expD)
        
        for j in range(self.J+1):
            cond_probs[j] = expD[j]/denom
        
        return cond_probs
        
    def simulate_shares(self, prices):
        """Takes an array of prices (J prices, no zero for OO)
           Returns simulation of market shares
        """

        totals = np.zeros(self.J+1)
        for i in range(self.NS):
            totals = np.add(totals, self.cond_choice_prob(prices,self.P0[i]))
        return totals/self.NS

    def cond_choice_derivs(self, prices, nu):
        """This function takes prices, unobservables,
        Returns matrix of cond. share/price derivatives"""
        
        F = self.cond_choice_prob(prices, nu)
        #Check that this is correct
        Dmu = [nu[0] for j in range(self.J+1)]

        cD = np.zeros((self.J+1, self.J+1))
        
        for j in range(self.J+1):
            for q in range(j+1):
                if j == q:
                    cD[j][q] = F[j]*(1-F[j])*Dmu[j]
                else:
                    cD[j][q] = -F[j]*F[q]*Dmu[q]
                    cD[q][j] = cD[j][q]
        return cD

    def simulate_derivs(self, prices):
        """Takes prices,
           Returns (simulated) matrix of share/price derivatives
        """
        
        totals = np.zeros((self.J+1, self.J+1))
        for i in range(self.NS):
            totals = np.add(totals, self.cond_choice_derivs(prices, self.P0[i]))
        
        return totals/self.NS
        

    def make_Delta(self, prices):
        """Takes prices,
           Returns Delta matrix, where:
           Delta_jr = -ds_r/dp_j if r,j are produced by same firm
                    = 0          otherwise
           NB: Delta is JxJ matrix, we omit OO since it is produced by
               no firm and the row/column of 0s would make Delta singular
        """
        
        derivs = self.simulate_derivs(prices)
        Delta = np.zeros((self.J, self.J))
        for firm in self.ownership:
            for j in firm:
                for r in firm:
                    Delta[j-1][r-1] = -derivs[j][r]
        return Delta

    def choose_prices(self, prices):
        """This takes prices of all products,
           Returns optimal prices for each product, given those prices
        """

        D = np.linalg.inv(self.make_Delta(prices))
        s = self.simulate_shares(prices)[1:] #Trim OO
        return self.MC + np.dot(D, s)

    def equilibrium(self, tol=1e-6, theta=0.1, maxiter=1000):
        """This returns the EQ prices for this market"""

        prices0 = 10*np.ones(self.J)
        prices1 = np.ones(self.J)
        
        iter = 0
        diff = 100
        
        while diff > tol and iter < maxiter:
            prices1 = self.choose_prices(prices0)
            #print(prices1)
            iter += 1
            diff  = np.linalg.norm(prices0 - prices1)
            prices0 = theta*prices1+(1-theta)*prices0

            if iter % 20 == 0:
                print(iter, " : ", diff)

        shares = self.simulate_shares(prices0)[1:] #Trim OO    
        return prices0, shares

    def produce_data(self, t, mkt):
        """Returns an array describing equilibrium prices, shares, etc. for each product"""
        prices, shares = self.equilibrium()

        Data = [[t, mkt, j, prices[j], shares[j]] for j in range(self.J)]
        for j in range(self.J):
            Data[j].extend(self.prod_chars[j])
            Data[j].extend(self.cost_chars[j])
            Data[j].append(self.MC[j])
        return Data

np.random.seed(0)
num_prod = 10
theta = [0.5, 3, 2.5, 2, 1.5, 1]       
beta  = [0.8, 1, 1.5, 2, 2.5, 3]
gamma = [0.5, 1]
prod_chars = np.random.rand(num_prod, 5)
cost_chars = np.random.rand(num_prod, 2)
ownership = [[1,2,3,4], 
             [5,6],
             [7,8,9],
             [10]]

Data = []
for t in range(10):
    marg_cost_t = np.exp(np.random.normal(size=num_prod)/4)
    for mkt in range(100):
        print("Computing: ", (t, mkt), "...")
        foo = Market(theta, beta, gamma, 
                     ownership, prod_chars, cost_chars,
                     marg_cost_t, NS=1000)
        Data.extend(foo.produce_data(t, mkt))

with open("test_data.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(Data)
