import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class Positions(Enum):
    Short = 0
    Long = 1
    Sideways = 2 

    def change_pos(self, pos):
        if pos == 0:
            return Positions.Short
        elif pos == 1:
            return Positions.Long
        else:
            return Positions.Sideways


class Wallet(object):
    def __init__(self, init_money) -> None:
        self.avgprice = 0
        self.leverage = 1
        self.n_contract = 0
        self.position = Positions.Sideways

        self.cash_balance = init_money
        self.total_balance = self.cash_balance

        self.unrealized_pl = 0
        self.realized_pl = 0


    def _update_wallet(self,current_price):
        if self.position == Positions.Long:
            self.unrealized_pl = (current_price - self.avgprice) * self.leverage * self.n_contract
        elif self.position == Positions.Short:
            self.unrealized_pl = (self.avgprice - current_price) * self.leverage * self.n_contract

        self.total_balance = self.cash_balance + self.unrealized_pl
    
    def sell_all(self, current_price):
        prev_pl = 0

        self._update_wallet(current_price)
        self.realized_pl = self.unrealized_pl 
        self.unrealized_pl = 0

        self.cash_balance += self.realized_pl

        prev_pl += self.realized_pl
        self.realized_pl = 0
        self.avgprice = 0
        self.n_contract = 0

        self._update_wallet(current_price)

        return prev_pl
    
    def add_contract(self, current_price, n_add_contact):
        prev_pl = 0

        if n_add_contact > 0:
            # buy
            self.avgprice = (self.avgprice*self.n_contract  + current_price*n_add_contact)/ (self.n_contract+n_add_contact)
            self.n_contract += n_add_contact
            
            self._update_wallet(current_price)

        elif n_add_contact < 0:
            # sell
            n_add_contact = abs(n_add_contact)
            if self.position == Positions.Long:
                self.realized_pl = (current_price - self.avgprice) * n_add_contact
            elif self.position == Positions.Short:
                self.realized_pl = (self.avgprice - current_price) * n_add_contact

            self.cash_balance += self.realized_pl

            prev_pl += self.realized_pl
            self.realized_pl = 0
            self.avgprice = self.avgprice
            self.n_contract -= n_add_contact

            self._update_wallet(current_price)

        return prev_pl

    def change_pos(self, to_pos):
        self.position = self.position.change_pos(to_pos)

        return self.position

class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size, trade_time, reward_scale, init_money):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.init_money = init_money
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.MultiDiscrete([len(Positions), 100]) # Predict only + portfolio management 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)
        self.reward_scale = reward_scale

        # episode
        if trade_time == None:
            self._start_tick = self.window_size
            self._end_tick = len(self.prices) - 1
        else:
            self._start_tick = np.random.randint(self.window_size, self.df.shape[0]-trade_time)
            self._end_tick = self._start_tick + trade_time
        self._done = None
        self._current_tick = None

        self.wallet = None
        self.total_reward = 0
        self.total_profit = 0

        self._first_rendering = None
        self.history = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        
        self.wallet = Wallet(self.init_money)
        self.total_reward = 0.
        self.total_profit = 0 
        self.percentage_profit = 0  
        self._first_rendering = True
        self.history = {}
        return self._get_observation()


    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick or self.wallet.unrealized_pl <= -self.wallet.cash_balance:
            self._done = True
            predict = Positions.Sideways
            n_contract = 0
        
        else:
            predict = action[0]
            n_contract = action[1]

        step_reward = self._cal_pl(predict=predict, n_want_contract=n_contract)
        self.total_reward += step_reward

        observation = self._get_observation()

        info = dict(
            total_reward = self.total_reward,
            total_profit = self.total_profit,
            percentage_profit = self.percentage_profit,
            position = self.wallet.position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info


    def render(self, mode='human'):
        plt.figure(figsize=(15,6))
        plt.cla()

        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        sideways_ticks = []
        last_tick = None
        for i in range(len(self.history["position"])):
            if self.history["position"][i]  == last_tick:
                pass

            elif self.history["position"][i] == Positions.Short.value:
                short_ticks.append(i+self._start_tick)
            elif self.history["position"][i] == Positions.Long.value:
                long_ticks.append(i+self._start_tick)
            elif self.history["position"][i] == Positions.Sideways.value:
                sideways_ticks.append(i+self._start_tick)

            last_tick = self.history["position"][i]

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')
        plt.plot(sideways_ticks, self.prices[sideways_ticks], 'yo')

        plt.suptitle(
            "Total Reward: %.4f" % self.total_reward + ' ~ ' +
            "Total profit: %.4f" % self.total_profit + ' ~ ' +
            "Total Percentage profit: %.4f" % self.percentage_profit + '%'
        )

        plt.show()


    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size):self._current_tick]


    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)



        
    def close(self):
        plt.close()


    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def _process_data(self):
        raise NotImplementedError


    def _cal_pl(self, action):
        raise NotImplementedError


    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError