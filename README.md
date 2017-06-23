# Multiview LL-LVM

This repository contains MATLAB code that implements the Multiview Locally Linear Latent Variable Model (MLL-LVM). 
MLL-LVM is a fully Bayesian model for manifold learning solved with [Variational Inference].
Observations are assumed to consist of multiple views or modes, i.e. different types of observations.
In this sense the model is 'multiview'.
Each view corresponds to a different mode of observation independent of the other views.
All views, albeit conditionally independent, share a single set of embedding coordinates.

After learning the model with Variational Inference, one view can be inferred based on knowledge of the other views. 

## Graphical Model

![Graphical Model](img/graphical_model.png?raw=true "MLL-LVM Graphical Model")

The training set is assumed to contain N observations, each consisting of V views.
Model parameters are either latent/probabilistic or deterministic/non-probabilistic.
* Latent variables
  * C : Local manifold tangents. A prior constrains tangents of neighbouring data to be close to one another.
  * x : Embedding coordinates. A prior constrains embeddings of neighbouring data to be close to one another.
  * y : Observations.
* Deterministic parameters
  * G : Data neighbourhood graph structure. 
  * γ : Controls the prior on C.
  * α : Controls the prior on x.

## Testing the model

In order to run the tests, change into the folder ```code/scripts/```.

### Learning the Swiss roll

The first two tests are run on the swiss-roll dataset.
Run the swiss-roll scripts with:

* ```swissroll_demo``` Tests the single-view LL-LVM code base on Swiss roll (see comments on LL-LVM at the end of this readme).
* ```swissroll_demo_multi_splitintotwoviews``` Tests on Swiss roll data after splitting it into V=2 views: The dimension for the one view is 2, and for the other view it is 1. The resulting embedding coordinates are similar to the single-view case.

### Estimation of missing views

For this test we shall assume a set of 3d points sampled off a helix, plus noise:

![Test manifold](img/est_demo_helix2.png?raw=true "Test manifold (Helix)")

Info about the points is given as two views. View 1 comprises of one of the point variates, leaving the other two variates for View 2.

Given a number of points for which only view 2 is known, we aim to estimate view 1.
The related test script can be run with:

* ```estimation_demo```

## Paper
The related paper has been presented in the 3<sup>rd</sup> workshop on Bayesian and Graphical Models in Biomedical Imaging (BAMBI), 
held in conjunction with the 19<sup>th</sup> International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI).

*The paper won the 'best paper award' of the workshop*.

Please cite this work as:

```bibtex
@InProceedings{Sfikas16Bayesian,
  author       = "G. Sfikas and C. Nikou",
  title        = "Bayesian Multiview Manifold Learning applied to Hippocampus Shape and Clinical Score Data",
  booktitle    = "Proceedings of the $3^{rd}$ International Workshop on Bayesian and Graphical Models in Biomedical Imaging, held in conjunction with MICCAI'16",
  year         = "2016"
}
```

You can download the paper at http://www.cs.uoi.gr/~sfikas/16Sfikas_MLL-LVM.pdf .


## Single-view model (LL-LVM)
The current model extends on the single-view locally-linear latent variable model [LL-LVM](http://arxiv.org/abs/1410.6791) presented in [NIPS 2015](https://nips.cc/Conferences/2015/AcceptedPapers).
The Matlab code for this work can be found [here](https://github.com/mijungi/lllvm/).

In a nutshell, the current MLL-LVM extends on LL-LVM in these aspects:
* Data points can be seen as sets of views. Likewise, LL-LVM can be considered a special case of MLL-LVM for number of views V=1. 
* Missing views can be inferred given observed views with a mechanism that derives from the model.

[OASIS cross-section dataset]: <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2895005/>
[Variational Inference]: <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.610.2116&rep=rep1&type=pdf>
