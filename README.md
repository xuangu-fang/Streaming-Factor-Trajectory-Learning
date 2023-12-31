# Streaming-Factor-Trajectory-Learning
This authors' official PyTorch implementation for paper: [Streaming Factor Trajectory Learning for Temporal Tensor Decomposition](https://arxiv.org/abs/2310.17021)(NeurIPS 2023) 


## Streaming Learning of Temporal Tensor Data

### Each factor is a time-varing and uncertainty-aware function
To model the temporal tensor data, we assign each tensor factor with Temporal-Gaussian Process priors and model it as the continuous-time-varing trajectory.
![model illustration](./figs/model.PNG)

### Online inference for streaming data: 
The inference of the factor trajectory is in a streaming and online manner.
![Online inference by running posterior update](./figs/streaming.PNG)

Example of learned functional frajectories of factors from real-world data.
<!-- ![Learned Factor Trajectories](./figs/learnt_functions.PNG) -->

<div align=center> <img src="./figs/learnt_functions.PNG" width = 80%/> </div>

## Requirements:
The project is mainly built with pytorch 1.10.1 under python 3. Besides that, make sure to install [tqdm](https://tqdm.github.io/) and [tensorly](http://tensorly.org/stable/index.html) before running the project.

## Instructions:
1. Clone this repository.
2. To play with the code quickly with visulization of factors, we offer several notebooks at `notebook`(on synthetic & real data)
3. To run the real-world datasets with scripts, see `my_script.sh` for example.
4. To tune the (hyper)parametrs of model, modify the `.yaml` files in `config` folder
5. To apply the model on your own dataset, please follow the `./data/process_script/beijing_15k_process.ipynb` or  `./data/synthetic/simu_data_generate_CP_r1.ipynb` to process the raw data into appropriate format.

Please check our paper for more details.

## Citation
Please cite our work if you would like it
```
@misc{fang2023streaming,
      title={Streaming Factor Trajectory Learning for Temporal Tensor Decomposition}, 
      author={Shikai Fang and Xin Yu and Shibo Li and Zheng Wang and Robert Kirby and Shandian Zhe},
      year={2023},
      eprint={2310.17021},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```