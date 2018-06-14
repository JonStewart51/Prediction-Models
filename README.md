# Prediction-Models

This project is for custom models. There are a few different models:

1) There are a minimum description length random forest and similarity forest models. The difference between these models and regular random forest and similarity forest models, is that the votes of the individual trees are not equal. Specifically, they are weighted using the minimum description length principle, when applied to decision tree based models, says that if two trees have similar training accuracy, then the tree with less depth is likely to be the more accurate on data it has not been trained on. There are accuracy weighted trees, where each tree's vote is weighted by the training accuracy, but these tend to overfit, while capping the maximum tree length to an arbitrarily low value tends to increase error due to bias. One way to approach this issue is to include both accuracy and tree length as inputs to the weight of each tree's vote. This adds two parameters to each model, and cross validation can be used to set the ratio at which accuracy and tree length parameters are each weighted. Since similarity forests obtain second order statistics, while random forests obtain first order statistics, the models can be complementary when used together.


2) There are several neural network models, written in Keras+tensorflow and mxnet. Some of these are written for R, and some with Python depending upon the preprocessing needed and other considerations. These models include:

- Wavenet:  Additionally, it's meant for predicting time series, not building sequences of speech as in the original paper.  Here, the gating is also x*sigmoid(x). The gating function for these is slightly different than in the original paper. This helps ensure that back propagation can still correct errors, since the derivative of x*sigmoid(x) is simply sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x)). That first term in the derivative makes error propagation easier, ensuring weights are accurately updated. This gating convention is borrowed from the paper,  "Swish, A self generated Activation Function". This is written in Python, using Keras.

