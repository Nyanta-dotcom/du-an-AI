# Data

Datasets are not committed to Git. Put them here:

```
data/
├── raw/
│   ├── sign_mnist/
│   │   ├── sign_mnist_train.csv
│   │   └── sign_mnist_test.csv
│   └── landmark/
│       ├── keypoint.csv
│       └── LICENSE_dataset_MIT.md
└── processed/            # created by: python -m src.data
```

| Dataset | Source | Used by |
|---|---|---|
| Sign Language MNIST (28×28 grayscale, 24 letters) | https://www.kaggle.com/datasets/datamunge/sign-language-mnist | `src/data.py` → CNN, baselines |
| Hand landmarks (21 × (x, y), MIT licence) | https://github.com/AkramOM606/American-Sign-Language-Detection (`keypoint.csv`) | `src/train_landmark_mlp.py` → web app |
