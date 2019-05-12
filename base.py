## Contains base interfaces for various algorithms.

from abc import ABC, abstractmethod

class RLAlgorithm(ABC):

	@abstractmethod
	def log(self, fname): pass
	# Start logging to file fname. Note that the file must be closed in the class destructor.

	def modify_reward(self, r): return r
	# Modify existing reward r to whatever is needed.

	@abstractmethod
	def trial(self): pass
	# Conduct a trial of multiple epochs.