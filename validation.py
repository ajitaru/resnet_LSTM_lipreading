# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

import torch, os, torch.optim as optim
from torch.autograd import Variable
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
from xinshuo_miscellaneous import print_log

class Validator():
    def __init__(self, options):
        self.validationdataset = LipreadingDataset(options["general"]["dataset"], "val", False)
        self.validationdataloader = DataLoader(self.validationdataset, batch_size=options["input"]["batchsize"], 
            shuffle=options["input"]["shuffle"], num_workers=options["input"]["numworkers"], drop_last=True)
        self.usecudnn = options["general"]["usecudnn"]
        self.batchsize = options["input"]["batchsize"]
        self.statsfrequency = options["training"]["statsfrequency"]
        self.gpuid = options["general"]["gpuid"]
        self.log_file = options["general"]["logfile"]
        print_log('loaded validation dataset with %d data' % len(self.validationdataset), log=self.log_file)

    def epoch(self, model):
        print_log("Starting validation...", log=self.log_file)
        count = 0
        validator_function = model.validator_function()
        for i_batch, sample_batched in enumerate(self.validationdataloader):
            input = Variable(sample_batched['temporalvolume'])
            labels = sample_batched['label']
            if(self.usecudnn):
                input = input.cuda(self.gpuid)
                labels = labels.cuda(self.gpuid)

            outputs = model(input)
            count += validator_function(outputs, labels)
            print_log(count, log=self.log_file)

        accuracy = count / len(self.validationdataset)
        accu_savepath = os.path.join(options["general"]["modelsavedir"], 'accuracy_epoch%03d.txt')
        with open(accu_savepath, "a") as outputfile:
            outputfile.write("\ncorrect count: {}, total count: {} accuracy: {}".format(count, len(self.validationdataset), accuracy))