# Prediction-Models

This project is for custom models. There are a few different models:

1) There are a minimum description length random forest and similarity forest models. The difference between these models and regular random forest and similarity forest models, is that the votes of the individual trees are not equal. Specifically, they are weighted using the minimum description length principle, when applied to decision tree based models, says that if two trees have similar accuracy, then the tree with less depth is likely to be the more accurate on data it has not been trained on. There are accuracy weighted trees, where each tree's vote is weighted by the accuracy, but these tend to overfit, while capping the maximum tree length to an arbitrarily low value tends to increase error due bias. One way to approach this issue is to include both accuracy and tree length as inputs to the weight of each tree's vote. This adds two parameters to each model, and cross validation can be used to set the ratio at which accuracy and tree length parameters are each weighted.


2) There are several neural network models, written in Keras+tensorflow and mxnet. Some of these are written for R, and some with Python. It depended upon the preprocessing needed and other considerations. 
