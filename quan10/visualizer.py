import matplotlib.pyplot as plt, numpy as np

class Visualizer:
    def __init__(self):
        self.E = []
        self.name = "<>"

    def set_name(self, name):
        self.name = name

    def append(self, energy):
        self.E.append(energy)

    def visualize(self):
        self.calc_energies()
        self.convergence_plot()
        self.calc_order_stats()
        self.print_order_stats()

    def convergence_plot(self, _plt=None):
        counts = np.arange(0, len(self.energies))
        values = self.energies
        
        if _plt != None:
            _plt.plot(counts, values, label=self.name)
        else:
            _plt = plt
            _plt.plot(counts, values)
            _plt.rcParams["figure.figsize"] = (12, 4)
            _plt.xlabel("Eval count")
            _plt.ylabel("Energy")
            _plt.title("Convergence plot for " + self.name)
            _plt.show()
            

    def calc_energies(self):
        self.energies = np.array(self.E)
        self.energies = self.energies.reshape(self.energies.size)
    
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
                orders.append((i, order_of_accuracy, new_order))
            i += 1
            
        self.order_stats = np.array(orders)
        
    def print_order_stats(self):
        array = self.order_stats.transpose()
        print('\n'.join([''.join(['{:8}'.format('{0:.2f}'.format(item)) for item in row]) 
              for row in array]))

    def order_plot(self, _plt=None):
        counts = self.order_stats[:,0]
        values = self.order_stats[:,1]
        if _plt != None:
            _plt.plot(counts, values, label=self.name)
        else:
            _plt = plt
            _plt.plot(counts, values)
            _plt.rcParams["figure.figsize"] = (12, 4)
            _plt.xlabel("Eval count")
            _plt.ylabel("Order of accuracy ($\\log_e (\\frac{\\Delta}{x_f})$)")
            _plt.title("Order of accuracy plot for " + self.name)
            _plt.show()


class MetaVisualizer:
    def __init__(self, seqs):
        self.seqs = seqs

    def convergence_plots(self):
        plt.rcParams["figure.figsize"] = (12, 4)
        plt.xlabel("Eval count")
        plt.ylabel("Energy")
        for seq in self.seqs:
            # yikes, (maybe) a violation of OOP
            seq.vis.calc_energies()
            seq.vis.convergence_plot(plt)
        plt.legend()
        plt.show()

    def order_plots(self):
        plt.rcParams["figure.figsize"] = (12, 4)
        plt.xlabel("Eval count")
        plt.ylabel("Order of accuracy ($\\log_e (\\frac{\\Delta}{x_f})$)")
        for seq in self.seqs:
            seq.vis.calc_energies()
            seq.vis.calc_order_stats()
            # print(seq.vis.order_stats[:,0])
            # print(seq.vis.order_stats[:,1])
            seq.vis.order_plot(plt)
        plt.legend()
        plt.show()