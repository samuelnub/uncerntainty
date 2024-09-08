import ROOT
import Network
import torch.optim as optim
import torch
import os.path
import numpy as np

if __name__ == '__main__':
    dataFilename = 'MasterclassData.root' #'tmva_class_example.root'


    rfile = ROOT.TFile.Open(f'./data/{dataFilename}')
    for key in rfile.GetListOfKeys():
        name = key.GetName()
        entries = rfile.Get(name).GetEntries()
        print('{} : {}'.format(name, entries))


    # Tree name within ROOT file to work with,
    # As well as independent and dependent variables

    treeName = 'DecayTree;1'

    ivNames = ['kaon_OWNPV_X', 'kaon_OWNPV_Y', 'kaon_OWNPV_Z']
    dvNames = ['pion_OWNPV_X', 'pion_OWNPV_Y', 'pion_OWNPV_Z']

    # Extract ROOT data to Numpy
    dataFrame = ROOT.RDataFrame(treeName, f'./data/{dataFilename}')
    ivDict: dict = dataFrame.AsNumpy(ivNames)
    dvDict: dict = dataFrame.AsNumpy(dvNames)

    print(type(ivDict))
    print(ivDict)
    print(dvDict)

    ivData: torch.Tensor | None = None
    dvData: torch.Tensor | None = None

    for i in range(len(ivDict[ivNames[0]])): # All IV's and DV's should have same no. of samples
        # Iterate over each column / key of IV and DV
        ivEntryArr = []
        dvEntryArr = []

        for ivName in ivNames:
            ivEntryArr.append(ivDict[ivName][i])
        ivTensor = torch.from_numpy(np.array(ivEntryArr))

        for dvName in dvNames:
            dvEntryArr.append(dvDict[dvName][i])
        dvTensor = torch.from_numpy(np.array(dvEntryArr))

        if ivData == None or dvData == None:
            ivData = ivTensor.expand(1,-1)
            dvData = dvTensor.expand(1,-1)
        else:
            ivData = torch.cat((ivData, ivTensor.expand(1,-1)), dim=0)
            dvData = torch.cat((dvData, dvTensor.expand(1,-1)), dim=0)


    print(ivData.size())

    inChannels = len(ivNames)
    outChannels = len(dvNames)

    model: Network.Network = Network.Network(inChannels=inChannels,
                                             outChannels=outChannels)
    
    optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    modelFilename = f'IV-{"-".join(ivNames)}-DV-{"-".join(dvNames)}.pth'

    modelFilepath = f'./model/{modelFilename}'

    if os.path.exists(modelFilepath):
        # This configuration of IVs and DVs has already been learnt
        model.load_state_dict(torch.load(modelFilepath))
    else:
        # Train fresh model
        running_loss = 0
        last_loss = 0

        print('GONNA TRAIN NOW WAHHHH')

        # Save at the end

    model.eval()
    