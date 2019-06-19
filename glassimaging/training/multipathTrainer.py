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

    @staticmethod
    def loadFromCheckpoint(loc):
        checkpoint = torch.load(loc, map_location=MultipathTrainer.getDevice())
        net = createModel(checkpoint['model_desc'])
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(net.parameters(), lr=MultipathTrainer.lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return MultipathTrainer(net, optimizer, scheduler)

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

    def trainWithLoader(self, torchloader, epochs, testloader=None, maxBatchesPerEpoch=None, testinterval=5, p_drop_interval=10):
        self.logger.info('Using cuda version {v}.'.format(v=torch.version.cuda))
        for e in range(0, epochs):
            self.logger.info('Epoch {e} started training.'.format(e=e))
            running_loss = 0.0
            num_items = 0
            for i_batch, sample_batched in enumerate(torchloader):
                img = sample_batched["data"]
                seg = sample_batched["seg"]
                loss = self.trainWithBatch(img, seg)
                running_loss += loss.item()
                num_items += img.size(0)
                if maxBatchesPerEpoch is not None and i_batch >= maxBatchesPerEpoch - 1:
                    break
            loss = running_loss / (i_batch + 1)
            self.scheduler.step(loss)
            self.noteLoss(loss, category="DuringTraining", epoch=e)
            if e % testinterval == 0:
                loss = self.getLossWithLoader(torchloader, maxBatches=maxBatchesPerEpoch)
                self.noteLoss(loss, category="Train", epoch=e)
                if testloader is not None:
                    testloss = self.getLossWithLoader(testloader, maxBatches=maxBatchesPerEpoch)
                    self.noteLoss(testloss, category="Test", epoch=e)
                    self.logger.info('Loss calculated for epoch {e}. Train loss is {loss}.'.format(e=e, loss=loss))
            if (e % p_drop_interval) == p_drop_interval - 1:
                self.doublePDrop()

    def doublePDrop(self):
        if not self.parallel and hasattr(self.net, 'p_drop'):
            l = self.net.p_drop * 2
            if l < 1:
                self.net.setPDrop(l)
                self.logger.info('Set p_loss to {}'.format(l))
        elif self.parallel and hasattr(self.net.module, 'p_drop'):
            l = self.net.module.p_drop * 2
            if l < 0.4:
                self.net.module.setPDrop(l)
                self.logger.info('Set p_loss to {}'.format(l))
