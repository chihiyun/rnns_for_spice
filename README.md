# Implementations of RNNs for SPiCe

This code was implemented for the dataset on the SPiCe competition.

C. Shibata and J. Heinz, "Predicting Sequential Data with LSTMs Augmented with Strictly 2-Piecewise Input Vectors", in proc. of ICGI 2016 (to app\
ear).


# USAGE
First of all, please make sure that python 2.7 and Chainer (http://chainer.org) are installed.
Also, check whether gpu (cuda) can be used or not.

1. Download the SPiCe dataset (http://spice.lif.univ-mrs.fr).
2. Put it (named "on-line/") in the root directory of the repository "rnn_for_spice".
3. Execute the following command to learn the RNN models:
   - python learn.py
    
   Then, the RNN models and the log files are created in the directories whose names are the problem numbers of the SPiCe competition.
   
   Since it takes a very long time (more than one week) to run on a computer, if you want to avoid it, please adjust the parameters related to the main loop in "learn.py".

4. To obtain statistics such as the average scores from the learned models, execute the following command:
   - python test_statistic.py
