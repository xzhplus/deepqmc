from deepqmc import Molecule, evaluate, train
from deepqmc.wf import PauliNet

mol = Molecule.from_name('LiH')
net = PauliNet.from_hf(mol).cuda()
train(net)
evaluate(net)