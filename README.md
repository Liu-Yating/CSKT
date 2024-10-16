# CLIP-based Synergistic Knowledge Transfer for Text-based Person Retrieval


## Highlights

The goal of this work is to design  a CLIP-based Synergistic Knowledge Transfer (CSKT) approach for Text-based Person Retrieval,  which ecollaboratively enhances the deep fusion of V-L feature representations, and thoroughly leverages the CLIP’s underlying capacities in rich knowledge and cross-modal alignment.

## Usage
### Requirements
we use single NVIDIA V100 32G GPU for training and evaluation. 
```
pytorch 1.12.1
torchvision 0.13.1
prettytable
easydict
```

### Prepare Datasets
Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)

Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```


## Training

```

python train.py \
--name baseline \
--img_aug \
--batch_size 128 \
--dataset_name $DATASET_NAME \
--loss_names 'sdm' \
--num_epoch 60 \
--root_dir '/data/lyw/dataset_reid' \
--lr 3e-4 \
--depth 12 \
--n_ctx 4
```

## Testing

```python
python test.py --config_file 'path/to/model_dir/configs.yaml'
```

## Citation
If you find this code useful for your research, please cite our paper.

```tex
@inproceedings{liu2024clip,
  title={CLIP-based Synergistic Knowledge Transfer for Text-based Person Retrieval},
  author={Liu, Yating and Li, Yaowei and Liu, Zimo and Yang, Wenming and Wang, Yaowei and Liao, Qingmin},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7935--7939},
  year={2024},
  organization={IEEE}
}
```

## Acknowledgments
Some components of this code implementation are adopted from [CLIP](https://github.com/openai/CLIP), [IRRA](https://github.com/anosorae/IRRA) and [MaPLE](https://github.com/muzairkhattak/multimodal-prompt-learning). We sincerely appreciate for their contributions.

