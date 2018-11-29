# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

from __future__ import print_function
import torch, toml, os
from models import LipRead
from training import Trainer
from validation import Validator
from xinshuo_miscellaneous import get_timestring, print_log
from xinshuo_io import mkdir_if_missing

print("Loading options...")
with open('options.toml', 'r') as optionsFile: options = toml.loads(optionsFile.read())
if options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]: torch.backends.cudnn.benchmark = True 
options["general"]["modelsavedir"] = os.path.join(options["general"]["modelsavedir"], 'trained_model_' + get_timestring()); mkdir_if_missing(options["general"]["modelsavedir"])
options["general"]["logfile"] = open(os.path.join(options["general"]["modelsavedir"], 'log.txt'), 'w')

print_log('saving to %s' % options["general"]["modelsavedir"], log=options["general"]["logfile"])

print_log('creating the model', log=options["general"]["logfile"])
model = LipRead(options)

print_log('loading model', log=options["general"]["logfile"])
if options["general"]["loadpretrainedmodel"]: model.load_state_dict(torch.load(options["general"]["pretrainedmodelpath"]))		#Create the model.
if options["general"]["usecudnn"]: model = model.cuda(options["general"]["gpuid"])		#Move the model to the GPU.

print_log('loading data', log=options["general"]["logfile"])
if options["training"]["train"]: trainer = Trainer(options)
if options["validation"]["validate"]: 
	validator = Validator(options)
	validator.epoch(model, epoch=0)

for epoch in range(options["training"]["startepoch"], options["training"]["endepoch"]):
	if options["training"]["train"]: trainer.epoch(model, epoch)
	if options["validation"]["validate"]: validator.epoch(model, epoch)
	# if options["testing"]["test"]:
	# 	tester = Tester(options)
	# 	tester.epoch(model)

options["general"]["logfile"].close()