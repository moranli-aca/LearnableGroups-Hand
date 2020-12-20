# LearnableGroups-Hand

The code for the paper **Exploiting Learnable Joint Groups for Hand Pose Estimation** (Accepted by AAAI2021).

[Paper](https://arxiv.org/abs/2012.09496)


### Reference
```
@misc{li2020exploiting,
      title={Exploiting Learnable Joint Groups for Hand Pose Estimation}, 
      author={Moran Li and Yuan Gao and Nong Sang},
      year={2020},
      eprint={2012.09496},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Usage 
The code is built on Python3 and Pytorch 1.6.0.

#### Install dependencies 
```bash
pip install -r requirements.txt
```
#### Run the code 
- evaluate on the RHD: 
```bash
python eval_RHD.py --data_dir 'your RHD_published_v2 dataset path'
```

  
