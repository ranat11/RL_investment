from .trading_env import TradingEnv, Positions
import pandas as pd

class ForexEnv(TradingEnv):

    def __init__(self, env_data, window_size = 5, env_type = "train", trade_time = None, reward_scale = 100, init_money=100):
        assert env_type == "train" or "eval" or "test"
        

        df = pd.read_csv(env_data)
        df.set_index('date', inplace=True)
        df_train = df.iloc[ 0 : int(df.shape[0]*0.7) ]
        df_eval  = df.iloc[ int(df.shape[0]*0.7) - window_size: int(df.shape[0]*0.9) ]
        df_test  = df.iloc[ int(df.shape[0]*0.9) - window_size: ]

        if env_type == "train":
            df = df_train
        elif env_type == "eval":
            df = df_eval
            trade_time = 300
        else:
            df = df_test
            trade_time = None

        super().__init__(df, window_size, trade_time, reward_scale, init_money)
        self.trade_fee = 0.0003  # unit


    def _process_data(self):
        prices = self.df.loc[:, 'close'].to_numpy()

        # Normalize
        df_norm = (self.df-self.df.min())/(self.df.max()-self.df.min())
        signal_features = df_norm.loc[:, :].to_numpy()
        
        return prices, signal_features

      
    def _cal_pl(self, predict, n_want_contract):
        realize_pl = 0
        current_price = self.prices[self._current_tick]

        if self.wallet.position != predict:
            # Change pos
            if self.wallet.position == Positions.Long:
                realize_pl +=  self.wallet.sell_all(current_price)
            elif self.wallet.position == Positions.Short:
                realize_pl += -self.wallet.sell_all(current_price)

            self.wallet.position = self.wallet.change_pos(predict)
            
        
        if predict != Positions.Sideways:         
            # adjust portfolio
            n_add_contact = n_want_contract - self.wallet.n_contract  
            realize_pl += self.wallet.add_contract(current_price, n_add_contact)

        # cal _total_profit per day
        self.total_profit += realize_pl
        self.percentage_profit = (self.total_profit/self.init_money) /((self._current_tick - self._start_tick)/5) * 100

        return realize_pl * self.reward_scale
        
