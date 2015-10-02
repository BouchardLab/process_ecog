#!/usr/bin/env python
import os, sys
from pylearn2.config import yaml_parse
import numpy as np
import whetlab

print 'Imports done...'

yaml = sys.argv[1]
print yaml
with open(yaml, 'r') as f:
    text = f.read()
print text
train = yaml_parse.load(text)
train.main_loop()
