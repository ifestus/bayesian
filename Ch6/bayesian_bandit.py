import numpy as np

class Bandits(object):
   def __init__(self, p_array):
      self.p = p_array
      self.optimal = np.argmax(self.p)

   def pull(self, i):
      return np.random.rand() < self.p[i]

   def __len__(self):
      return len(self.p)

class BayesianStrategy(object):
   def __init__(self, bandits):
      self.bandits = bandits
      n_bandits = len(self.bandits)
      self.wins = np.zeros(n_bandits)
      self.trials = np.zeros(n_bandits)
      self.N = 0
      self.choices = []
      self.bb_score = []

   def sample(self, n=1):
      bb_score = np.zeros(n)
      choices = np.zeros(n)

      for k in range(n):
         choice = np.argmax(np.random.beta(1+self.wins, 1+self.trials-self.wins))

         result = self.bandits.pull(choice)

         self.wins[choice] += result
         self.trials[choice] += 1
         self.N += 1
         choices[k] = choice
         bb_score[k] = result

      self.bb_score = np.r_[self.bb_score, bb_score]
      self.choices = np.r_[self.choices, choices]

def main():
   hidden_probs = np.array([0.85, 0.60, 0.75])
   bandits = Bandits(hidden_probs)
   bayesian_strat = BayesianStrategy(bandits)

   draw_samples = [1, 1, 3, 10, 10, 25, 50, 100, 200, 600, 1000, 2500]

   for i in enumerate(draw_samples):
      bayesian_strat.sample()

   wins = bayesian_strat.wins
   trials = bayesian_strat.trials

   print(np.random.beta(1+wins, 1+trials-wins))

if __name__ == '__main__':
   main()
