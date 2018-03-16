# mc-dropout-mnist
Implementation of (parts of) the experiment on MNIST from [Bayesian Convolutional Neural Networks with
Bernoulli Approximate Variational Inference](http://mlg.eng.cam.ac.uk/yarin/PDFs/NIPS_2015_bayesian_convnets.pdf)

Standard LeNet architecture without Dropout is compared against a LeNet-all architecture, where Dropout is applied after each layer (including convolutions).
Dropout is kept at test time, and the prediction of the trained model is averaged over *T=50* stochastic passes. 
The MC-Dropout model achieves an error rate of ~0.6%, compared to ~1% of the non-dropout model.

Required libraries:
`tqdm, keras`

Tested with Tensorflow and Python 3.
