# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

from torch.autograd import Variable
import torch, torch.optim as optim, os, math
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
from xinshuo_miscellaneous import print_log
from xinshuo_io import fileparts

def timedelta_string(timedelta):
    totalSeconds = int(timedelta.total_seconds())
    hours, remainder = divmod(totalSeconds,60*60)
    minutes, seconds = divmod(remainder,60)
    return "{} hrs, {} mins, {} secs".format(hours, minutes, seconds)

def output_iteration(i, time, totalitems, log_file):
    avgBatchTime = time / (i+1)
    estTime = avgBatchTime * (totalitems - i)
    print_log("Iteration: {}\nElapsed Time: {} \nEstimated Time Remaining: {}".format(i, timedelta_string(time), timedelta_string(estTime)), log=log_file)

class Trainer():
    def __init__(self, options):
        self.batchsize = options["input"]["batchsize"]
        self.trainingdataset = LipreadingDataset(options["general"]["dataset"], "train")
        self.trainingdataloader = DataLoader(self.trainingdataset, batch_size=self.batchsize,
            shuffle=options["training"]["shuffle"], num_workers=options["input"]["numworkers"], drop_last=True)
        self.usecudnn = options["general"]["usecudnn"]
        self.statsfrequency = options["training"]["statsfrequency"]
        self.gpuid = options["general"]["gpuid"]
        self.learningrate = options["training"]["learningrate"]
        # self.modelType = options["training"]["learningrate"]
        self.weightdecay = options["training"]["weightdecay"]
        self.momentum = options["training"]["momentum"]
        self.log_file = options["general"]["logfile"]
        self.modelsavedir = options["general"]["modelsavedir"]
        _, self.time_str, _ = fileparts(self.modelsavedir)
        print_log('loaded training dataset with %d data' % len(self.trainingdataset), log=options["general"]["logfile"])

    def learningRate(self, epoch):
        decay = math.floor((epoch - 1) / 5)
        return self.learningrate * pow(0.5, decay)

    def epoch(self, model, epoch):
        #set up the loss function.
        criterion = model.loss()
        optimizer = optim.SGD(model.parameters(), lr=self.learningRate(epoch), momentum=self.learningrate, weight_decay=self.weightdecay)

        #transfer the model to the GPU.
        if self.usecudnn: criterion = criterion.cuda(self.gpuid)
        startTime = datetime.now()
        print_log("Starting training...", log=self.log_file)
        for i_batch, sample_batched in enumerate(self.trainingdataloader):
            optimizer.zero_grad()
            input = Variable(sample_batched['temporalvolume'])
            labels = Variable(sample_batched['label'])
            if(self.usecudnn):
                input = input.cuda(self.gpuid)
                labels = labels.cuda(self.gpuid)

            outputs = model(input)
            loss = criterion(outputs, labels.squeeze(1))
            print_log('Training: {}, Epoch: {}, loss is {}'.format(self.time_str, epoch, loss.item()), log=self.log_file)
            loss.backward()
            optimizer.step()
            sampleNumber = i_batch * self.batchsize
            if sampleNumber % self.statsfrequency == 0:
                currentTime = datetime.now()
                output_iteration(sampleNumber, currentTime - startTime, len(self.trainingdataset), self.log_file)

        print_log("Epoch completed, saving state...", log=self.log_file)
        torch.save(model.state_dict(), os.path.join(self.modelsavedir, 'trained_model_epoch%03d.pt' % epoch))