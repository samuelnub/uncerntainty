import ROOT

if __name__ == '__main__':
    datafilename = 'tmva_class_example'


    rfile = ROOT.TFile.Open(f'./data/{datafilename}.root')
    for key in rfile.GetListOfKeys():
        name = key.GetName()
        entries = rfile.Get(name).GetEntries()
        print('{} : {}'.format(name, entries))


    treename = 'DecayTree;1'

    ivnames = []
    dvnames = []
