import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import deque

class SingleAssetCPPI:
    def __init__(self, data, start_date, end_date):
        self.data = data
        self.start_idx = data.index.get_loc(start_date)
        self.end_idx = data.index.get_loc(end_date)
        self.period = self.end_idx - self.start_idx

        self.date = data[self.start_idx:self.end_idx+1].index
        self.raw = 10000 * data[self.start_idx:self.end_idx+1].values.T[0] / data.loc[start_date][0]

        self.daily_return = self.raw[1:]/self.raw[:-1] -1

    def reset(self):
        self.current_idx = 0

    def run(self, DEPTH=0.15, PARAM=3, LOOKBACK_WINDOW=250, VOL_WINDOW=100, vol_method='exp'):
        self.INITIAL_PRICE = 10000
        self.PARAMETER = PARAM
        self.LOOKBACK_WINDOW =LOOKBACK_WINDOW
        self.VOL_WINDOW = VOL_WINDOW

        price = self.INITIAL_PRICE
        histry = deque([self.INITIAL_PRICE]*self.LOOKBACK_WINDOW, maxlen=self.LOOKBACK_WINDOW)
        pre_data = self.data[self.start_idx-VOL_WINDOW-1:self.start_idx].values.T[0]
        return_data = deque(pre_data[1:]/pre_data[:-1] - 1, maxlen=VOL_WINDOW)
        volatility = self.get_vol(return_data)
        multiplier = 1 / (self.PARAMETER * volatility)
        floor = np.max(histry) * (1 - DEPTH)
        ratio = np.maximum(0, np.minimum(1.0, multiplier * (price - floor)/price))

        results = np.zeros((self.period+1,5))
        self.reset()
        results[self.current_idx] = np.array([price,
                                              volatility,
                                              floor,
                                              ratio,
                                              self.raw[self.current_idx]])
        
        while True:
            r = self.daily_return[self.current_idx]
            return_data.append(r)
            price = price * (1 + ratio * r)
            histry.append(price)
            tmp_ratio = ratio*(1+r)/(1+ratio*r)

            volatility = self.get_vol(return_data, method=vol_method)
            multiplier = 1 / (self.PARAMETER * volatility)
            floor = np.max(histry) * (1 - DEPTH)
            ratio = np.maximum(0, np.minimum(1.0, multiplier * (price - floor)/price))
            if np.abs(ratio-tmp_ratio)<0.05:
                ratio = tmp_ratio

            self.current_idx += 1
            results[self.current_idx] = np.array([price,
                                                 volatility,
                                                 floor,
                                                 ratio,
                                                 self.raw[self.current_idx]])

            if self.current_idx==self.period:
                break
        
        return results.T


    def get_vol(self, return_data, method='exp', HL=30):
        if method=='exp':
            weights = [np.exp(np.log(0.5)*t/HL) for t in range(self.VOL_WINDOW,0,-1)]
            weights = weights / np.sum(weights)
            diff = return_data - np.mean(return_data)
            vol = np.sqrt(np.sum(weights * diff**2)*250)
        elif method=='normal':
            vol = np.std(return_data)*np.sqrt(250)
        return vol

class SingleAssetOBPI:
    def __init__(self, data, start_date, end_date):
        self.data = data
        self.start_idx = data.index.get_loc(start_date)
        self.end_idx = data.index.get_loc(end_date)
        self.period = self.end_idx - self.start_idx

        self.date = data[self.start_idx:self.end_idx+1].index
        self.raw = 10000 * data[self.start_idx:self.end_idx+1].values.T[0] / data.loc[start_date][0]

        self.daily_return = self.raw[1:]/self.raw[:-1] -1

    def reset(self):
        self.current_idx = 0

    def run(self, DEPTH=0.15, PARAM=3, LOOKBACK_WINDOW=250, VOL_WINDOW=100, vol_method='exp'):
        self.INITIAL_PRICE = 10000
        self.PARAMETER = PARAM
        self.LOOKBACK_WINDOW =LOOKBACK_WINDOW
        self.VOL_WINDOW = VOL_WINDOW

        price = self.INITIAL_PRICE
        histry = deque([self.INITIAL_PRICE]*self.LOOKBACK_WINDOW, maxlen=self.LOOKBACK_WINDOW)
        pre_data = self.data[self.start_idx-VOL_WINDOW-1:self.start_idx].values.T[0]
        return_data = deque(pre_data[1:]/pre_data[:-1] - 1, maxlen=VOL_WINDOW)
        volatility = self.get_vol(return_data)
        floor = np.max(histry) * (1 - DEPTH)
        remaining_term = np.maximum(10, np.argmax(reversed(histry)))/250
        BS = BlackScholes(price, 0, remaining_term, floor, 0, volatility, 0)
        multiplier = price*BS.call_delta()/BS.call_premium()
        ratio = np.maximum(0, np.minimum(1.0, multiplier * (price - floor)/price))

        results = np.zeros((self.period+1,5))
        self.reset()
        results[self.current_idx] = np.array([price,
                                              volatility,
                                              floor,
                                              ratio,
                                              self.raw[self.current_idx]])
        
        while True:
            r = self.daily_return[self.current_idx]
            return_data.append(r)
            price = price * (1 + ratio * r)
            histry.append(price)
            tmp_ratio = ratio*(1+r)/(1+ratio*r)

            volatility = self.get_vol(return_data, method=vol_method)
            floor = np.max(histry) * (1 - DEPTH)
            remaining_term = np.maximum(10, np.argmax(reversed(histry)))/250
            BS = BlackScholes(price, 0, remaining_term, floor, 0, volatility, 0)
            multiplier = price*BS.call_delta()/BS.call_premium()
            ratio = np.maximum(0, np.minimum(1.0, multiplier * (price - floor)/price))
            if np.abs(ratio-tmp_ratio)<0.05:
                ratio = tmp_ratio

            self.current_idx += 1
            results[self.current_idx] = np.array([price,
                                                 volatility,
                                                 floor,
                                                 ratio,
                                                 self.raw[self.current_idx]])

            if self.current_idx==self.period:
                break
        
        return results.T 

    def get_vol(self, return_data, method='exp', HL=30):
        if method=='exp':
            weights = [np.exp(np.log(0.5)*t/HL) for t in range(self.VOL_WINDOW,0,-1)]
            weights = weights / np.sum(weights)
            diff = return_data - np.mean(return_data)
            vol = np.sqrt(np.sum(weights * diff**2)*250)
        elif method=='normal':
            vol = np.std(return_data)*np.sqrt(250)
        return vol   

class SingleAssetBullSpread:
    def __init__(self, data, start_date, end_date):
        self.data = data
        self.start_idx = data.index.get_loc(start_date)
        self.end_idx = data.index.get_loc(end_date)
        self.period = self.end_idx - self.start_idx

        self.date = data[self.start_idx:self.end_idx+1].index
        self.raw = 10000 * data[self.start_idx:self.end_idx+1].values.T[0] / data.loc[start_date][0]

        self.daily_return = self.raw[1:]/self.raw[:-1] -1

    def reset(self):
        self.current_idx = 0

    def run(self, FLOORDEPTH=0.15, CAPDEPTH=0.15, PARAM=3, LOOKBACK_WINDOW=250, VOL_WINDOW=100, vol_method='exp', strategy='point_change'):
        self.INITIAL_PRICE = 10000
        self.PARAMETER = PARAM
        self.LOOKBACK_WINDOW =LOOKBACK_WINDOW
        self.VOL_WINDOW = VOL_WINDOW

        price = self.INITIAL_PRICE
        pre_data = self.data[self.start_idx-VOL_WINDOW-1:self.start_idx].values.T[0]
        return_data = deque(pre_data[1:]/pre_data[:-1] - 1, maxlen=VOL_WINDOW)
        volatility = self.get_vol(return_data)
        floor = price * (1 - FLOORDEPTH)
        cap = price * (1 + CAPDEPTH)
        changepoint = (floor + cap) * 0.5
        remaining_term = np.maximum(10, LOOKBACK_WINDOW)/250
        if strategy=='point_change':
            greeks = np.zeros((self.period+1, 2))
            if price <= changepoint:
                BS = BlackScholes(price, 0, remaining_term, floor, 0, volatility, 0)
                greeks[0] = np.array([BS.call_delta(), BS.call_premium()])
                multiplier = price*BS.call_delta()/BS.call_premium()
                ratio = np.maximum(0.1, np.minimum(1.0, multiplier * (price - floor)/price))
            else:
                BS = BlackScholes(price, 0, remaining_term, cap, 0, volatility, 0)
                greeks[0] = np.array([BS.put_delta(), BS.put_premium()])
                multiplier = price*BS.put_delta()/BS.put_premium()
                ratio = np.maximum(0.1, np.minimum(1.0, multiplier * (price - cap)/price))
        elif strategy=='integration':
            greeks = np.zeros((self.period+1, 4))
            upper = BlackScholes(price, 0, remaining_term, cap, 0, volatility, 0)
            lower = BlackScholes(price, 0, remaining_term, floor, 0, volatility, 0)
            greeks[0] = np.array([lower.call_delta(), upper.call_delta(), lower.call_premium(), upper.call_premium()])
            multiplier = price*(lower.call_delta()-upper.call_delta())/(lower.call_premium()-upper.call_premium())
            ratio = np.maximum(0.1,np.minimum(1.0, multiplier * (price - floor)/price))
        
        results = np.zeros((self.period+1,5))
        self.reset()
        results[self.current_idx] = np.array([price,
                                              volatility,
                                              floor,
                                              ratio,
                                              self.raw[self.current_idx]])
        
        while True:
            r = self.daily_return[self.current_idx]
            return_data.append(r)
            price = price * (1 + ratio * r)
            tmp_ratio = ratio*(1+r)/(1+ratio*r)

            self.current_idx += 1
            volatility = self.get_vol(return_data, method=vol_method)
            if self.current_idx % LOOKBACK_WINDOW == 0:
                floor = price * (1 - FLOORDEPTH)
                cap = price * (1 + CAPDEPTH)
                changepoint = (floor + cap) * 0.5
            remaining_term = np.maximum(10, LOOKBACK_WINDOW - self.current_idx % LOOKBACK_WINDOW)/250
            if strategy=='point_change':
                if price <= changepoint:
                    BS = BlackScholes(price, 0, remaining_term, floor, 0, volatility, 0)
                    greeks[self.current_idx] = np.array([BS.call_delta(), BS.call_premium()])
                    multiplier = price*BS.call_delta()/BS.call_premium()
                    ratio = np.maximum(0.1, np.minimum(1.0, multiplier * (price - floor)/price))
                else:
                    BS = BlackScholes(price, 0, remaining_term, cap, 0, volatility, 0)
                    greeks[self.current_idx] = np.array([BS.put_delta(), BS.put_premium()])
                    multiplier = price*BS.put_delta()/BS.put_premium()
                    ratio = np.maximum(0.1, np.minimum(1.0, multiplier * (price - cap)/price))
            elif strategy=='integration':
                upper = BlackScholes(price, 0, remaining_term, cap, 0, volatility, 0)
                lower = BlackScholes(price, 0, remaining_term, floor, 0, volatility, 0)
                greeks[self.current_idx] = np.array([lower.call_delta(), upper.call_delta(), lower.call_premium(), upper.call_premium()])
                multiplier = price*(lower.call_delta()-upper.call_delta())/(lower.call_premium()-upper.call_premium())
                ratio = np.maximum(0.1,np.minimum(1.0, multiplier * (price - floor)/price))
            if np.abs(ratio-tmp_ratio)<0.05:
                ratio = tmp_ratio

            results[self.current_idx] = np.array([price,
                                                 volatility,
                                                 floor,
                                                 ratio,
                                                 self.raw[self.current_idx]])

            if self.current_idx==self.period:
                break
        
        return np.concatenate([results.T, greeks.T], axis=0)

    def get_vol(self, return_data, method='exp', HL=30):
        if method=='exp':
            weights = [np.exp(np.log(0.5)*t/HL) for t in range(self.VOL_WINDOW,0,-1)]
            weights = weights / np.sum(weights)
            diff = return_data - np.mean(return_data)
            vol = np.sqrt(np.sum(weights * diff**2)*250)
        elif method=='normal':
            vol = np.std(return_data)*np.sqrt(250)
        return vol   

class BlackScholes:
    def __init__(self, S_t, t, T, K, r, sigma, q=0):
        self.S_t = S_t
        self.t = t
        self.T = T
        self.K = K
        self.r = r
        self.sigma = sigma
        self.q = q

        self.d1 = (np.log(self.S_t/self.K)+(self.r-self.q+0.5*self.sigma**2)*(self.T-self.t))/(self.sigma*np.sqrt(np.maximum(1e-10,self.T-self.t)))
        self.d2 = (np.log(self.S_t/self.K)+(self.r-self.q-0.5*self.sigma**2)*(self.T-self.t))/(self.sigma*np.sqrt(np.maximum(1e-10,self.T-self.t)))

    def call_premium(self):
        CP = np.exp(-self.q*(self.T-self.t))*self.S_t*ss.norm.cdf(self.d1) - self.K*np.exp(-self.r*(self.T-self.t))*ss.norm.cdf(self.d2)
        return CP
    def put_premium(self):
        PP = -np.exp(-self.q*(self.T-self.t))*self.S_t*ss.norm.cdf(-self.d1) + self.K*np.exp(-self.r*(self.T-self.t))*ss.norm.cdf(-self.d2)
        return PP
    
    def call_delta(self):
        CD = np.exp(-self.q*(self.T-self.t))*ss.norm.cdf(self.d1)
        return CD
    def put_delta(self):
        PD = np.exp(-self.q*(self.T-self.t))*(ss.norm.cdf(self.d1) - 1)
        return PD
    
    def call_gamma(self):
        CG = np.exp(-self.q*(self.T-self.t))*ss.norm.pdf(self.d1) / (self.S_t*self.sigma*np.sqrt(np.maximum(1e-10,self.T-self.t)))
        return CG
    def put_gamma(self):
        PG = np.exp(-self.q*(self.T-self.t))*ss.norm.pdf(self.d1) / (self.S_t*self.sigma*np.sqrt(np.maximum(1e-10,self.T-self.t)))
        return PG
    
    def call_vega(self):
        CV = self.S_t*np.exp(-self.q*(self.T-self.t))*ss.norm.pdf(self.d1)*np.sqrt(np.maximum(1e-10,self.T-self.t))
        return CV
    def put_vega(self):
        PV = self.S_t*np.exp(-self.q*(self.T-self.t))*ss.norm.pdf(self.d1)*np.sqrt(np.maximum(1e-10,self.T-self.t))
        return PV
    
    def call_theta(self):
        CT = (-self.S_t*self.sigma*np.exp(-self.q*(self.T-self.t))*ss.norm.pdf(self.d1))/(2*np.sqrt(np.maximum(1e-10,self.T-self.t))) + self.q*self.S_t*np.exp(-self.q*(self.T-self.t))*ss.norm.cdf(self.d1) - self.r*self.K*np.exp(-self.r*(self.T-self.t))*ss.norm.cdf(self.d2)
        return CT
    def put_theta(self):
        PT = (-self.S_t*self.sigma*np.exp(-self.q*(self.T-self.t))*ss.norm.pdf(self.d1))/(2*np.sqrt(np.maximum(1e-10,self.T-self.t))) - self.q*self.S_t*np.exp(-self.q*(self.T-self.t))*ss.norm.cdf(-self.d1) + self.r*self.K*np.exp(-self.r*(self.T-self.t))*ss.norm.cdf(-self.d2)
        return PT

    def call_rho(self):
        CR = self.K*(self.T-self.t)*np.exp(-self.r*(self.T-self.t))*ss.norm.cdf(self.d2)
        return CR
    def put_rho(self):
        PR = -self.K*(self.T-self.t)*np.exp(-self.r*(self.T-self.t))*ss.norm.cdf(-self.d2)
        return PR

    def call_greeks(self):
        d = {"premium":self.call_premium(),
             "delta":self.call_delta(),
             "gamma":self.call_gamma(),
             "vega":self.call_vega(),
             "theta":self.call_theta(),
             "rho":self.call_rho()}
        return d
    def put_greeks(self):
        d = {"premium":self.put_premium(),
             "delta":self.put_delta(),
             "gamma":self.put_gamma(),
             "vega":self.put_vega(),
             "theta":self.put_theta(),
             "rho":self.put_rho()}
        return d

class Simulation:
    def __init__(self, seed=0, path_num=100, term=250, block_length=30):
        self.SEED = seed
        self.PATH_NUM = path_num
        self.TERM = term
        self.BLOCK_LENGTH = block_length

    def make_senario(self, return_data, method="GBM"):
        self.return_data = np.array(return_data)

        ## Geometric Brownian Motion
        if method == "GBM":
            np.random.seed(self.SEED)
            mu = np.mean(self.return_data)
            sigma = np.std(self.return_data)
            scenario = mu + sigma * np.random.randn(self.PATH_NUM, self.TERM)
        
        ## bootstrap
        elif method == "bootstrap":
            np.random.seed(self.SEED)
            candidate_num = len(self.return_data)
            pick_idx = np.random.randint(0, candidate_num, (self.PATH_NUM, self.TERM))
            scenario = self.return_data[pick_idx]
        
        ## Moving Block Bootstrap
        elif method == "MBB":
            np.random.seed(self.SEED)
            candidate_num = len(self.return_data) - self.BLOCK_LENGTH
            block_num = int(np.ceil(self.TERM/self.BLOCK_LENGTH))
            rnd_idx = np.random.randint(0, candidate_num, (self.PATH_NUM, block_num))
            pick_idx = np.array([elem + np.arange(self.BLOCK_LENGTH) for row in rnd_idx for elem in row])
            pick_idx = pick_idx.reshape(self.PATH_NUM,-1)
            pick_idx = pick_idx[:,:self.TERM]
            scenario = self.return_data[pick_idx]
        
        return scenario
    
    def run_simulate(self, return_data, strategy="cppi", method="GBM", DEPTH=0.15, PARAM=1, vol_method="exp", bull_strategy="integrate"):
        self.method = method
        self.return_data = np.array(return_data)
        self.DEPTH = DEPTH
        self.PARAM = PARAM
        self.vol_method = vol_method
        self.bull_strategy = bull_strategy

        scenario = self.make_senario(self.return_data, method=self.method)
        pre_scenario = np.tile(self.return_data[-100:], (self.PATH_NUM,1))
        scenario = np.concatenate((pre_scenario, scenario), axis=1)
        price_data = 10000 * np.cumprod(np.insert(1 + scenario, 0, 1, axis=1), axis=1)
        end_idx = scenario.shape[1]
        start_idx = end_idx - self.TERM + 1
        paths = np.zeros((self.PATH_NUM, self.TERM))

        for num in range(self.PATH_NUM):
            if strategy=="cppi":
                cppi = SingleAssetCPPI(pd.DataFrame(price_data[num]), start_idx, end_idx)
                price, volatility, floor, ratio, raw = cppi.run(DEPTH=self.DEPTH, PARAM=self.PARAM, vol_method=self.vol_method)
            elif strategy=="obpi":
                obpi = SingleAssetOBPI(pd.DataFrame(price_data[num]), start_idx, end_idx)
                price, volatility, floor, ratio, raw = obpi.run(DEPTH=self.DEPTH, PARAM=self.PARAM, vol_method=self.vol_method)
            elif strategy=="bull_spred":
                if bull_strategy=="integrate":
                    bull = SingleAssetBullSpread(pd.DataFrame(price_data[num]), start_idx, end_idx)
                    price, volatility, floor, ratio, raw, lower_delta, upper_delta, lower_premium, upper_premium = bull.run(FLOORDEPTH=self.DEPTH, CAPDEPTH=self.DEPTH, PARAM=self.PARAM, vol_method=self.vol_method, strategy=self.bull_strategy)
                elif bull_strategy=="point_change":
                    bull = SingleAssetBullSpread(pd.DataFrame(price_data[num]), start_idx, end_idx)
                    price, volatility, floor, ratio, raw, delta, premium = bull.run(FLOORDEPTH=self.DEPTH, CAPDEPTH=self.DEPTH, PARAM=self.PARAM, vol_method=self.vol_method, strategy=self.bull_strategy)
            paths[num,:] = price
        return paths

def get_performance(data):
    price = np.array(data)
    r = price[1:]/price[:-1] - 1
    mu = np.mean(r)*250
    sigma = np.std(r)*np.sqrt(250)
    sharpe_ratio = mu / sigma
    max_DD = np.max(get_rolldown(price))
    return mu, sigma, sharpe_ratio, max_DD

def get_rolldown(data):
    price = np.array(data)
    rolldown = np.zeros(len(price)-250)
    for i in range(len(price)-250):
        bottom = np.min(price[i+1:i+250])
        rolldown[i] = 1 - bottom/price[i]
    return rolldown
