import ROOT
import Network
import torch.optim as optim
import os.path

if __name__ == '__main__':
    dataFilename = 'MasterclassData' #'tmva_class_example'


    rfile = ROOT.TFile.Open(f'./data/{dataFilename}.root')
    for key in rfile.GetListOfKeys():
        name = key.GetName()
        entries = rfile.Get(name).GetEntries()
        print('{} : {}'.format(name, entries))


    treename = 'DecayTree;1'

    ivnames = []
    dvnames = []

    inChannels = len(ivnames)
    outChannels = len(dvnames)

    model: Network.Network = Network.Network(inChannels=inChannels,
                                             outChannels=outChannels)
    
    optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    modelFilename = f'IV-{'-'.join(ivnames)}-DV-{'-'.join(dvnames)}.pth'

    
    