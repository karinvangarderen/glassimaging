import torch.optim as optim
import torch
from glassimaging.models.utils import createModel
from glassimaging.training.standardTrainer import StandardTrainer


class MultipathTrainer(StandardTrainer):

    def __init__(self, net, optimizer, scheduler=None):
        super(MultipathTrainer, self).__init__(net, optimizer, scheduler)

    @staticmethod
    def initFromDesc(desc):
        net = createModel(desc)
        optimizer = optim.Adam(net.parameters(), lr=MultipathTrainer.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return MultipathTrainer(net, optimizer, scheduler=scheduler)

    def loadExistingModels(self, model_locations):
        state_dicts = []
        for loc in model_locations:
            checkpoint = torch.load(loc, map_location=MultipathTrainer.getDevice())
            state_dicts = state_dicts + [checkpoint['model_state_dict']]
        if self.parallel:
            self.net.module.loadExistingModels(state_dicts)
        else:
            self.net.loadExistingModels(state_dicts)

    def trainLastLayerOnly(self):
        if self.parallel:
            self.net.module.trainLastLayerOnly()
        else:
            self.net.trainLastLayerOnly()
