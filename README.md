# NOPA: Neurally-guided Online Probabilistic Assistance for Building Socially Intelligent Home Assistants

This is the official implementation of the paper [*NOPA: Neurally-guided Online Probabilistic Assistance for Building Socially Intelligent Home Assistants*](https://arxiv.org/abs/2301.05223). 


In this work, we study how to build socially intelligent robots to assist people in their homes. In particular, we focus on assistance with online goal inference, where robots must simultaneously infer humans' goals and how to help them achieve those goals. For that, we propose **NOPA** (Neurally-guided Online Probabilistic Assistance), a method that predicts a set of goals given some observed human behavior and assist the human by taking actions based on the uncertainty of the inferred goal. 

To test this framework, we compare our method against multiple baselines in a new embodied AI assistance challenge: Online Watch-And-Help, in which a helper agent needs to simultaneously watch a main agent's action, infer its goal, and help perform a common household task faster in realistic virtual home environments.
 
 
![](assets/cover_fig_final.png | width=150)

We provide a dataset of tasks to evaluate the challenge, as well as different baselines consisting on learning and planning-based agents.

Check out a video of the work [here](https://youtu.be/Oawo9pynPL0).

## Cite
If you use this code in your research, please consider citing.

```
@misc{https://doi.org/10.48550/arxiv.2301.05223,
  doi = {10.48550/ARXIV.2301.05223},
  url = {https://arxiv.org/abs/2301.05223},
  author = {Puig, Xavier and Shu, Tianmin and Tenenbaum, Joshua B. and Torralba, Antonio},
  keywords = {Robotics (cs.RO), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), Multiagent Systems (cs.MA), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {NOPA: Neurally-guided Online Probabilistic Assistance for Building Socially Intelligent Home Assistants},
  publisher = {arXiv},
  year = {2023}
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Setup
### Get the VirtualHome Simulator and API
Clone the [VirtualHome API](https://github.com/xavierpuigf/virtualhome.git) repository one folder above this repository

```bash
cd ..
git clone https://github.com/xavierpuigf/virtualhome.git
cd virtualhome
pip install -r requirements.txt
```

Download the simulator, and put it in an `executable` folder, one folder above this repository


- [Download](http://virtual-home.org/release/simulator/v2.0/linux_exec.zip) Linux x86-64 version.
- [Download](http://virtual-home.org/release/simulator/v2.0/macos_exec.zip) Mac OS X version.
- [Download](http://virtual-home.org/release/simulator/windows_exec.zip) Windows version.

### Install Requirements
```bash
pip install -r requirements.txt
```



## Dataset
Dataset coming soon



## Test the NOPA model
Coming soon

## Visualize results
Coming soon