# Driving-IRL-NGSIM

## Getting started
Install the dependent package
```shell
pip install -r requirements.txt
```

Download the NGSIM dataset from [this website](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj) (export to csv file) and run dump_data.py along with the path to the downloaded csv file (may take a while)
```shell
python dump_data.py [YOUR PATH]/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv
```



## Reference
```
@article{huang2021driving,
  title={Driving Behavior Modeling Using Naturalistic Human Driving Data With Inverse Reinforcement Learning},
  author={Huang, Zhiyu and Wu, Jingda and Lv, Chen},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2021},
  publisher={IEEE}
}
```

## License
This repo is released under the MIT License. The NGSIM data processing code is borrowed from [NGSIM interface](https://github.com/Lemma1/NGSIM-interface). The NGSIM env is built on top of [highway env](https://github.com/eleurent/highway-env) which is released under the MIT license.
