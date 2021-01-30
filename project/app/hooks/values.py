import os,sys


class values:

	def __init__(self, loss, time, micro, macro):
		self.loss = loss
		self.time = time
		self.micro = micro
		self.macro = macro

    def get_loss(self):
        return self.loss

    def get_time(self):
        return self.time

    def get_micro_score(self):
        return self.micro	

    def get_macro_score(self):
        return self.macro
		
    def set_loss(self, loss):
        self.loss = loss

    def set_time(self, time):
        self.time = time

    def set_micro_score(self, micro):
        self.micro = micro	

    def set_macro_score(self, macro):
        self.macro = macro