import torch.utils.data
from data.nyuv2_dataset_crop import NYUDataset, NYUDataset_val


def CreateDataset(opt):

    dataset = NYUDataset()
    if opt.vallist != '':
        dataset_val = NYUDataset_val()
    else:
        dataset_val = None
    dataset.initialize(opt)
    if dataset_val != None:
        dataset_val.initialize(opt)

    return dataset, dataset_val

class CustomDatasetDataLoader():

    def initialize(self, opt):
        self.dataset, self.dataset_val = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
        if self.dataset_val != None:
            self.dataloader_val = torch.utils.data.DataLoader(
                self.dataset_val,
                batch_size=1,
                shuffle=False,
                num_workers=int(opt.nThreads))
        else:
            self.dataloader_val = None

    def load_data(self):
        return self.dataloader, self.dataloader_val

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
