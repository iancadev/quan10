# if all we do are basic qiskit statevectors, qiskit ansatze,
#      sparsepauli hamiltonian observables, and scipy minimize
#      then just remove half the methods and it will look clean
# the only reason it looks complicated is I want to try experimenting
#      with pennylane ansatze and pennylane optimizers
# there probably isn't time, but the interface won't change much.
# the logic tree for incorporating different backends, circuits,
#      measuring schemes was always going to be complicated unless
#      we want to go down a fully procedural route

# it might also look better if we have functions that take in any type of
# arg, but then use if statements to route the arg based on type
# rather than having methods with the desired type in their name

# design question: generate the ansatz 





from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit import QuantumRegister, QuantumCircuit

import sys
for k in ['visualizer', 'Visualizer']:
    if k in sys.modules: del sys.modules[k]
import visualizer
from visualizer import Visualizer


class Sequencer:
    """ ORDER: program, backend, ansatz """
    
    def __init__(self):
        self.structure = {
            'backend': None,
            'ansatz': None,
            'hamiltonian': None,
            'output': None
        }
        self.vis = Visualizer()

    def does(self, args):
        """ Confirm that the Sequencer has a structure consistent with args """
        for key in args:
            if key not in self.structure or args[key] != self.structure[key]:
                return False
        return True

    def set_name(self, name):
        self.name = name
        self.vis.set_name(name)

    # --- Backend : setting, init'ing, casting --- #
    
    def set_backend(self, name):
        name = name.lower()
        if name == 'statevector':
            self.structure['backend'] = 'qiskit'
            self.make_backend_statevector()
        else:
            print(name, 'not supported')

    def make_backend_statevector(self):
        self.statevector_estimator = StatevectorEstimator()
        self.statevector_sampler = StatevectorSampler()

    # --- Ansatz --- #
    
    def set_ansatz(self, name):
        if name == 'qiskit-EfficientSU2':
            self.structure['ansatz'] = 'qiskit'
            self.structure['ansatz-type'] = 'EfficientSU2'
            self.init_ansatz_qiskit(EfficientSU2)
        else:
            print(name, 'not supported')

    def init_ansatz_qiskit(self, circuit):
        self.ansatz = circuit(self.program.size)
    
    def make_ansatz(self):
        if self.structure['backend'] == self.structure['ansatz'] == 'qiskit':
            if self.structure['ansatz-type'] == 'EfficientSU2':
                circuit = EfficientSU2
            return self.ansatz # circuit(self.program.size)

    # --- Hamiltonian --- #

    def set_program(self, program):
        self.program = program
        if getattr(program, 'ham_sparseop', None):
            self.structure['hamiltonian'] = 'sparseop'
        else:
            print('hamiltonian program not supported')

        if getattr(program, 'interpret_qiskit_result', None):
            self.structure['output'] = 'sample'

    # --- Cost functions --- #

    def cost_func(self, params):
        if self.does({
            'backend': 'qiskit',
            'ansatz': 'qiskit',
            'hamiltonian': 'sparseop',
        }):
            return self.cost_func_sve(params)
        else:
            print('program not supported')

    def cost_func_sve(self, params):
        qreg = QuantumRegister(self.program.size)
        qc = QuantumCircuit(qreg)
        ans = self.make_ansatz()
        qc.compose(ans, inplace=True)
        
        obs = self.program.ham_sparseop()
        estimator = StatevectorEstimator()
        job = self.statevector_estimator.run([(qc, obs, params)])
        result = job.result()
    
        energy_cost = self.program.cost_qiskit_result(result)
        self.vis.append(energy_cost)
        
        return energy_cost

    # --- Final eval functions --- #
    
    def ansatz_eval(self, params):
        if self.does({
            'backend': 'qiskit',
            'ansatz': 'qiskit',
            'hamiltonian': 'sparseop',
            'output': 'sample'
        }):
            return self.ansatz_eval_sample(params)

    def ansatz_eval_sample(self, params):
        ans = self.make_ansatz()
        ans.measure_all()
        shots = self.program.size * 1000

        pub = (ans, params)
        job = self.statevector_sampler.run([pub], shots=shots)
        return job.result()[0]

    # --- extra stuff --- #

    def visualize(self):
        self.vis.visualize()