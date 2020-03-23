# DC4
requires sklearn, xgboost, numpy, pandas, scipy, and matplotlib

run idea2.py on the command line with: 
                                       py idea2.py (NAME OF QUORA TSV FILE) (windows) 
                                       OR 
                                       python idea2.py (NAME OF QUORA TSV FILE) (mac)

there are several status updates throughout the process of the file running, and the process of training the model takes about 10 min.
Once the model has been trained, precision, recall, and f1 scores are printed to the screen, and a graph appears.
X out of the graph, and you are prompted to enter two questions, in which the xgb model determines if they are duplicate.
