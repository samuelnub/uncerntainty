import ROOT
import Network
import torch.optim as optim
import torch
import os.path

if __name__ == '__main__':
    dataFilename = 'MasterclassData.root' #'tmva_class_example.root'


    rfile = ROOT.TFile.Open(f'./data/{dataFilename}')
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

    modelFilepath = f'./model/{modelFilename}'

    if os.path.exists(modelFilepath):
        # This configuration of IVs and DVs has already been learnt
        model.load_state_dict(torch.load(modelFilepath))
    else:
        # Train fresh model
        
    
    