# GRU-Net: Gaussian attention aided dense skip connection based MultiResUNet for Breast Histopathology Image Segmentation
This is the official implementation  of "GRU-Net: Gaussian attention aided dense skip connection based MultiResUNet for Breast Histopathology Image Segmentation" (MIUA 2024).

### Overall workflow:
![architecture](https://github.com/AyushRoy2001/GRU-Net/assets/94052139/63f5e08d-24ea-4516-a849-fb7204d2a954)

#### Controlled Dense Residual Block (CDRB)
![CDRB](https://github.com/AyushRoy2001/GRU-Net/assets/94052139/65c685a1-cc59-421e-ac22-39c212d50e94)

#### Gaussian distribution-based Attention Module (GdAM)
![GdAM](https://github.com/AyushRoy2001/GRU-Net/assets/94052139/3d4aa5b4-91c5-4896-82f7-47679ffdf2a9)

## Results
### Comparison with other methods
![motivation](https://github.com/AyushRoy2001/GRU-Net/assets/94052139/c40a57af-e736-4832-9b25-ccd5a3b3008f)

### GdAM 
#### TNBC
![TNBC_GdAM](https://github.com/AyushRoy2001/GRU-Net/assets/94052139/48bbe33c-7289-4436-925f-a9cd9e31ab14)

#### Monuseg
![MonuSeg_GdAM](https://github.com/AyushRoy2001/GRU-Net/assets/94052139/5c9e2349-235d-4872-8fd7-a5ce8e135428)

### Encoder and Decoder layers
#### TNBC
![TNBC_heat](https://github.com/AyushRoy2001/GRU-Net/assets/94052139/e5c62da4-e80b-4c30-a168-4f70bf0e5065)

#### Monuseg
![MonuSeg_heat](https://github.com/AyushRoy2001/GRU-Net/assets/94052139/1858d278-b8db-4970-bf2c-9e283aae5d18)

## Authors :nerd_face:
Ayush Roy<br/>
Payel Pramanik<br/>
Soham Ghosal<br/>
Daria Valenkova<br/>
Dmitrii Kaplun<br/>
Ram Sarkar<br/>

## Citation :thinking:
Please do cite our paper in case you find it useful for your research.<br/>
```
@inproceedings{roy2024gru,
  title={GRU-Net: Gaussian Attention Aided Dense Skip Connection Based MultiResUNet for Breast Histopathology Image Segmentation},
  author={Roy, Ayush and Pramanik, Payel and Ghosal, Sohom and Valenkova, Daria and Kaplun, Dmitrii and Sarkar, Ram},
  booktitle={Annual Conference on Medical Image Understanding and Analysis},
  pages={300--313},
  year={2024},
  organization={Springer}
}
```
