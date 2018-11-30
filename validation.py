# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

import torch, os, torch.optim as optim
from torch.autograd import Variable
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
from xinshuo_miscellaneous import print_log
from xinshuo_io import fileparts

class Validator():
    def __init__(self, options):
        self.batchsize = options["input"]["batchsize"]
        self.validationdataset = LipreadingDataset(options["general"]["dataset"], "val", False)
        self.validationdataloader = DataLoader(self.validationdataset, batch_size=self.batchsize, 
            shuffle=False, num_workers=options["input"]["numworkers"], drop_last=True)
        self.usecudnn = options["general"]["usecudnn"]
        self.statsfrequency = options["training"]["statsfrequency"]
        self.gpuid = options["general"]["gpuid"]
        self.log_file = options["general"]["logfile"]
        self.savedir = options["general"]["modelsavedir"]
        self.num_batches = int(len(self.validationdataset) / self.batchsize)
        print_log('loaded validation dataset with %d data' % len(self.validationdataset), log=self.log_file)

    def epoch(self, model, epoch):
        print_log("Starting validation...", log=self.log_file)
        count = 0
        validator_function = model.validator_function()
        for i_batch, (sample_batched, filename_batch) in enumerate(self.validationdataloader):
            with torch.no_grad():
                input = Variable(sample_batched['temporalvolume'])
                labels = sample_batched['label']
                if(self.usecudnn):
                    input = input.cuda(self.gpuid)
                    labels = labels.cuda(self.gpuid)        # num_batch x 1

                outputs = model(input)                      # num_batch x 500 for temp-conv         num_batch x 29 x 500               
                count_tmp, predict_index_list = validator_function(outputs, labels)
                count += count_tmp

                for batch_index in range(self.batchsize):
                    filename_tmp = filename_batch[batch_index]
                    _, filename_tmp, _ = fileparts(filename_tmp)
                    filename_tmp = filename_tmp.split('_')[0]
                    prediction_tmp = self.validationdataset.label_list[predict_index_list[batch_index]]
                    print_log('Evaluation: val set, batch index %d/%d, filename: %s, prediction: %s' % (batch_index+1, self.batchsize, filename_tmp, prediction_tmp), log=self.log_file)

                print_log('Evaluation: val set, batch %d/%d, correct so far %d/%d' % (i_batch+1, self.num_batches, count, self.batchsize*(i_batch+1)), log=self.log_file)

        accuracy = count / len(self.validationdataset)
        accu_savepath = os.path.join(self.savedir, 'accuracy_epoch%03d.txt' % epoch)
        print_log('saving the accuracy file to %s' % accu_savepath, log=self.log_file)
        with open(accu_savepath, "a") as outputfile:
            outputfile.write("\ncorrect count: {}, total count: {} accuracy: {}".format(count, len(self.validationdataset), accuracy))