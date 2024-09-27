import pylab, numpy as np

class Visualizer:
    def __init__(self):
        self.E = []
        self.name = "<>"

    def set_name(self, name):
        self.name = name

    def append(self, energy):
        self.E.append(energy)

    def visualize(self):
        if hasattr(self.E[0], '__iter__'):
            self.energies = np.array(self.E)
            self.energies = self.energies.reshape(self.energies.size)
        
        self.convergence_plot()
        self.calc_order_stats()
        self.print_order_stats()

    def convergence_plot(self):
        pylab.rcParams["figure.figsize"] = (12, 4)
        counts = np.arange(0, len(self.energies))
        values = [i for i in self.energies]
        pylab.plot(counts, values)
        pylab.xlabel("Eval count")
        pylab.ylabel("Energy")
        pylab.title("Convergence plot for " + self.name)

    def calc_order_stats_old(self):
        N = 5
        moving_average = np.convolve(self.energies, np.ones(N)/N, mode='valid')

        final = moving_average[-1]
        val = moving_average[-2]
        total_change = np.log(np.abs((val - final) / final))
        orders = []
        i = len(moving_average) - 1
        # print(final)
        for val in moving_average[-2::-1]:
            i -= 1
            if val != final:
                change_in_order = np.log(np.abs((val - final) / final)) - total_change
                # print(val, change_in_order, total_change)
                if change_in_order > 1:
                    total_change += change_in_order
                    orders.append((i, change_in_order, total_change))
        array = np.array(orders).transpose()
        self.order_stats = array[:,::-1]

    def calc_order_stats(self):
        N = 5
        moving_average = np.convolve(self.energies, np.ones(N)/N, mode='valid')

        final = moving_average[-1]
        val = moving_average[0]
        order_of_accuracy = -np.log(np.abs( (val - final) / final ))
        num = 0

        i = 1
        orders = []

        for val in moving_average[1:]:
            if val == final: continue
            new_order = -np.log(np.abs( (val - final) / final ))
            if new_order > order_of_accuracy + 1:
                order_of_accuracy += 1
                orders.append((i, new_order))
            i += 1
            
        self.order_stats = np.array(orders)
        
    
    def print_order_stats(self):
        array = self.order_stats.transpose()
        print('\n'.join([''.join(['{:8}'.format('{0:.2f}'.format(item)) for item in row]) 
              for row in arrays]))