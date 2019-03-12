# Algorithms for Optimization Jupyter Notebooks

This repository contains supplemental Jupyter notebooks to accompany [Algorithms for Optimization](http://mitpress.mit.edu/books/algorithms-optimization) by Mykel Kochenderfer and Tim Wheeler.
These notebooks were generated from the Algorithms for Optimization source code.
We provide these notebooks to aid with the development of lectures and understanding the material, with the hope that you find it useful.

## Installation
All notebooks have Julia 1.0.1 kernels.
[Julia can be installed here.](https://julialang.org/downloads/)

Rendering is managed by [PGFPlots.jl](https://github.com/JuliaTeX/PGFPlots.jl).
Please see [their documentation](https://nbviewer.jupyter.org/github/JuliaTeX/PGFPlots.jl/blob/master/doc/PGFPlots.ipynb) for important installation instructions.

Once the repo is cloned, one can set up the required packages from the terminal before launching the jupyter notebook:
```
export JULIA_PROJECT="@."
julia -e 'using Pkg; pkg"instantiate"'
jupyter notebook
```