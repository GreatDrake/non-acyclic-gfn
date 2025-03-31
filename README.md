# Revisiting Non-Acyclic GFlowNets in Discrete Environments

Official code for the paper [Revisiting Non-Acyclic GFlowNets in Discrete Environments](https://arxiv.org/abs/2502.07735).

Nikita Morozov*, Ian Maksimov*, Daniil Tiapkin, Sergey Samsonov.

## Abstract
Generative Flow Networks (GFlowNets) are a family of generative models that learn to sample objects from a given probability distribution, potentially known up to a normalizing constant. Instead of working in the object space, GFlowNets proceed by sampling trajectories in an appropriately constructed directed acyclic graph environment, greatly relying on the acyclicity of the graph. In our paper, we revisit the theory that relaxes the acyclicity assumption and present a simpler theoretical framework for non-acyclic GFlowNets in discrete environments. Moreover, we provide various novel theoretical insights related to training with fixed backward policies, the nature of flow functions, and connections between entropy-regularized RL and non-acyclic GFlowNets, which naturally generalize the respective concepts and theoretical results from the acyclic setting. In addition, we experimentally re-examine the concept of loss stability in non-acyclic GFlowNet training, as well as validate our own theoretical findings.

## Installation

```
git clone https://github.com/GreatDrake/non-acyclic-gfn.git
```

```
export PYTHONPATH="${PYTHONPATH}:./non-acyclic-gfn"
```

For the sake of reproducibility and stability, we highly recommend running the code on the newest PyTorch release.

## Hypergrids

Example of training with `SDB` loss in `flow` scale:

```
python run.py --dim 4 --side 20 --loss StableDB --loss_scale Flow --reg_coef 0.0 --name sdb --save_dir results
```

Example of training with `DB` loss in `log-flow` scale and `state flow regularization` with strength $\lambda = 0.001$:

```
python run.py --dim 4 --side 20 --loss DB --loss_scale LogFlow --reg_coef 0.001 --name db --save_dir results
```

## Permutations 

Example of training with `SDB` loss in `flow` scale:

```
python run_perms.py --p 8 --loss SDB --loss_scale Flow --reg_coef 0.0 --name db_example --save_dir example_dir
```

Example of training with `DB` loss in `log-flow` scale and `state flow regularization` with strength $\lambda = 0.001$:

```
python run_perms.py --p 8 --loss DB --loss_scale LogFlow --reg_coef 0.001 --name db_example --save_dir example_dir
```

For the full list of arguments for some particular script please refer to the script itself. 

## Citation

```
@article{morozov2025revisiting,
  title={Revisiting Non-Acyclic GFlowNets in Discrete Environments},
  author={Morozov, Nikita and Maksimov, Ian and Tiapkin, Daniil and Samsonov, Sergey},
  journal={arXiv preprint arXiv:2502.07735},
  year={2025}
}
```

