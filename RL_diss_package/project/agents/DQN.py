# NEED TO HANDLE EXCEPTIONS

class dqn():
    def __init__(self):
        pass

    def train(self):
        raise(NotImplementedError)

    def save_gif(self):
        raise(NotImplementedError)

    def save_state(self):
        # call for NN and RB to save their state as well
        raise(NotImplementedError)

    def display_progress(self):
        raise(NotImplementedError)

    def save_state(self):
        # call to NN and RB as well
        raise(NotImplementedError)

    def load_state(self):
        # load to NN and RB as well
        raise(NotImplementedError)
