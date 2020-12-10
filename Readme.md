# An Efficient Asynchronous Method for Integrating Evolutionary and Gradient-based Policy Search

This repository is the official implementation of [An Efficient Asynchronous Method for Integrating Evolutionary and Gradient-based Policy Search](https://papers.nips.cc/paper/2020/file/731309c4bb223491a9f67eac5214fb2e-Paper.pdf). 
> This repository is a reimplemented version for the public because the original source code is hard to follow.  
There may be an un-fixed bug in the reimplementation process.  
If you find one, please leave the issue. 

## Requirements
Python: 3.6.10  
pytorch: 1.1.0


To install requirements:
```bash
# Create conda environment
conda create -n aesrl python=3.6.10
conda activate aesrl

# CUDA 9.0
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch

# Cuda 10.0
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

pip install -r requirements.txt
```
## Initialize ray
Before the training process, ray framework should be initialized.  
You should run the initialization script on all machines.   
You can have one head machine which manages the training process, and other machines which only provide the resources.  
You may specify the port and password for the Redis-server.  
> We just used `12345` for both port and password for convenience. For exmple: `--redis-port 12345 --redis-password 12345`.

### Head
```bash
ray start --head --resources='{"machine": 1, "head": 1}' --port $PORT --redis-password $PASSWORD

# Example
ray start --head --resources='{"machine": 1, "head": 1}' --port 12345 --redis-password 12345
```
### Other machines
Before you start ray, you should export `PYTHONPATH`.  
`$PWD` should be the source directory.  
```bash
export PYTHONPATH=$PWD
```
`$HEADIP` is the IP address for the head node.  
```bash
ray start --address='$HEADIP:$PORT' --redis-password='$PASSWORD' --resources='{"machine": 1}'
```

## Train
We provide default hyperparameters for each algirithms in `config`.   
All hyperparameters are defined in `utils/util.py`

Replace $ENV_NAME with environment name in lower case.   
For example, `halfcheetah-v2/td3.json`
 

### Baselines

#### TD3
```bash
python train.py --config $ENV_NAME/td3.json --ray_address $HEADIP --ray_port $PORT --redis_password $PASSWORD
```
#### CEM-RL
This is our implementation of original CEM-RL in Serial-Synchronous scheme. 
```bash
python train.py --config $ENV_NAME/cemrl.json  --ray_address $HEADIP --ray_port $PORT --redis_password $PASSWORD
```
#### ACEM-RL
ACEM-RL is an asynchronous version of CEM-RL based on the previous work [1]  
You can specify the number of the actors with parameter `--num_critic_worker` and `--num_actor_worker`.
```bash
python train.py --config $ENV_NAME/acemrl.json --num_critic_worker 1 --num_actor_worker 5  \ 
--ray_address $HEADIP --ray_port $PORT --redis_password $PASSWORD
```
#### (1+1)-ES
Simple (1+1)-ES with 1/5 success rule. 
```bash
python train.py --config $ENV_NAME/opo.json --num_critic_worker 1 --num_actor_worker 5  \ 
--ray_address $HEADIP --ray_port $PORT --redis_password $PASSWORD
```
#### P-CEM-RL
This is parallel version of CEM-RL.  
```bash
python train.py --config $ENV_NAME/pcemrl.json --num_critic_worker 1 --num_actor_worker 5  \ 
--ray_address $HEADIP --ray_port $PORT --redis_password $PASSWORD
```
To train with parallel critic, 
```bash
python train.py --config $ENV_NAME/pcemrl.json --num_critic_worker 1 --num_actor_worker 5 --parallel-critic  \
--ray_address $HEADIP --ray_port $PORT --redis_password $PASSWORD
```

### Proposed algorithms
The algorithms below use several workers.   
You can specify the number of the actors with parameter `--num_critic_worker` and `--num_actor_worker`.
The final performance is measured with 1 critic worker and 5 actor workers.  

#### AES-RL
There are four mean update and two variance update algorithms.   
You can specify the update rules with the parameter `--aesrl_mean_update` and `--aesrl_var_update`.  
`--aesrl_mean_update` is one of [`fixed-linear`, `fixed-sigmoid`, `baseline-absolute`, `baseline-relative`].  
`--aesrl_var_update` is one of [`fixed`, `adaptive`].

You should specify the population ratio of RL and ES individuals in `--aesrl_rl_ratio`.   
Also, AES-RL always use the parallel critic.
Currently, multiple critic workers are not supported.  
All values in the original paper is measured with one critic worker. 

##### Mean update
We use `--aesrl_mean_update_param` for the parameter in the mean update.  
This parameter is a kind of environment specific reward normalization parameter, because the AES-RL uses fitness value itself.  
`aesrl_mean_update_param` is represented in the paper.  
For "baseline-absolute", this parameter is different from others. Therefore there are separate config files with postfix `_absolute`.
 
##### Variance update
When the variance update rule is `fixed`, you should specify the `n` with `--aesrl_fixed_var_n`.
In the original experiments, it `10` is used.

```bash
python train.py --config $ENV_NAME/aesrl.json --num_critic_worker 1 --num_actor_worker 5 \
--aesrl_mean_update $MEAN_UPDATE --aesrl_var_update $VAR_UPDATE  \
--aesrl_mean_update_param $PARAM [--aesrl_fixed_var_n $N]
--ray_address $HEADIP --ray_port $PORT --redis_password $PASSWORD
```

To get the final performance, `--aesrl_mean_update=baseline-relative`, `--aesrl_var_update=adaptive`
```bash
python train.py --config $ENV_NAME/aesrl.json --num_critic_worker 1 --num_actor_worker 5 \
--aesrl_mean_update baseline-relative --aesrl_var_update adaptive  --aesrl_mean_update_param $PARAM \
--ray_address $HEADIP --ray_port $PORT --redis_password $PASSWORD
```


## Reference
[1] Tobias Glasmachers. A natural evolution strategy with asynchronous strategy updates. In GECCO’13
