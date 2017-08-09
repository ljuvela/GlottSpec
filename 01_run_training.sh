#!/bin/bash

# toy example
python main.py --mode train --data ./traindata/ --rnn_context_len=1 --max_files=2

# proper training  example
#python main.py --mode train --data ./traindata/ --rnn_context_len=64 --max_files=30
