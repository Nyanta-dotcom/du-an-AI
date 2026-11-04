| Model | Features | Val acc | Val macro-F1 | Test acc | Test macro-F1 | Val-test gap | Train (s) | ms/img |
|---|---|---|---|---|---|---|---|---|
| Majority class (floor) | raw | 0.043 | 0.003 | 0.020 | 0.002 | +0.022 | 0.0 | 0.000 |
| Logistic Regression (pixels) | raw | 0.743 | 0.742 | 0.699 | 0.682 | +0.044 | 7.0 | 0.002 |
| Random Forest (pixels) | raw | 0.849 | 0.846 | 0.810 | 0.792 | +0.038 | 10.4 | 0.018 |
| MLP 256-128 (pixels) | raw | 0.837 | 0.837 | 0.760 | 0.735 | +0.077 | 7.3 | 0.004 |
| Linear SVM (HOG) | hog | 0.939 | 0.937 | 0.886 | 0.880 | +0.053 | 160.6 | 0.009 |
| MLP 256-128 (HOG) | hog | 0.930 | 0.930 | 0.906 | 0.899 | +0.024 | 14.0 | 0.005 |
