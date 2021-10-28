from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:


        # total_profit22 = self.training_env.get_attr("total_profit")[0]

        # if self.locals.get("dones")[0]:
        total_reward = self.training_env.buf_infos[0]['total_reward']
        total_profit = self.training_env.buf_infos[0]['total_profit']
        # percentage_profit = self.training_env.buf_infos[0]['percentage_profit']

        self.logger.record('rollout/total_reward', total_reward)
        self.logger.record('rollout/total_profit', total_profit)
        # self.logger.record('rollout/percentage_profit', percentage_profit)
    
        return True
