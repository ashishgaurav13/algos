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

	def args_k(self, *args): return [self.constants[item] for item in args]
	# Return a list of constants from keys

	def kwargs_k(self, *args): return {item: self.constants[item] for item in args}
	# Return a dict of constants from keys