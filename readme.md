# Indonesian Word Normalization Implementation

## Overview
This repo contains the implementation of word normalization using character-level seq2seq approach.
The transformer arcitecture from Vaswani et.al. (2017) is implemented on Colloquial Indonesian Lexicon from Salsabila et.al. (2018) and IndoCollex dataset from Wibowo et.al. (2021).
The dataset is taken from those sources then transformed into json files containing the informal-formal word pairs. The dataset containing those combined sources are also created.
Since the Colloquial Indonesian Lexicon only contains one file, the dataset is splitted into train, valid, test set with 80:10:10 proportion. 

## Running Script

```bash:
 python run_experiment.py --dataset_name <dataset_name> --num_epoch 200 --config_name <config_name> --model_name <output_model_name>
```

The `dataset_name` parameters can be filled with this following inputs
- `indocollex` for IndoCollex dataset
- `col_id_norm` for Colloquial Indonesian Lexicon dataset
- `combined` for combined dataset from IndoCollex and Colloquial Indonesian Lexicon

The `config_name` can be filled using the file name from the `./config` folder. The config folder contains configuration for the transformers architecture from
- The original "Attention is All You Need" paper form Vaswani et.al. (2017)
- The 'smaller transformer' implemented by Wu et.al. (2021) on "Applying the Transformer to Character-level Transduction" with different dropout value

## The original dataset repo

- [IndoCollex](https://github.com/haryoa/indo-collex)
- [Colloquial Indonesian Lexicon dataset](https://github.com/nasalsabila/kamus-alay)

## Cited Research
If you use this implementation please kindly cited these researches
- Aliyah Salsabila, N., Ardhito Winatmoko, Y., Akbar Septiandri, A., Jamal, A., 2018. Colloquial Indonesian Lexicon, in: 2018 International Conference on Asian Language Processing (IALP). Presented at the 2018 International Conference on Asian Language Processing (IALP), pp. 226–229. https://doi.org/10.1109/IALP.2018.8629151
- Wibowo, H.A., Nityasya, M.N., Akyürek, A.F., Fitriany, S., Aji, A.F., Prasojo, R.E., Wijaya, D.T., 2021. IndoCollex: A Testbed for Morphological Transformation of Indonesian Colloquial Words, in: Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021. Presented at the Findings 2021, Association for Computational Linguistics, Online, pp. 3170–3183. https://doi.org/10.18653/v1/2021.findings-acl.280
- Wu, S., Cotterell, R., Hulden, M., 2021. Applying the Transformer to Character-level Transduction, in: Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume. Presented at the EACL 2021, Association for Computational Linguistics, Online, pp. 1901–1907. https://doi.org/10.18653/v1/2021.eacl-main.163
- My research is comming soon (hopefully) 😅
