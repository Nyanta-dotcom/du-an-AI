| Model | Features | Val acc | Val macro-F1 | Test acc | Test macro-F1 | Val-test gap | Train (s) | ms/img |
|---|---|---|---|---|---|---|---|---|
| Majority class (floor) | raw | 0.047 | 0.004 | 0.020 | 0.002 | +0.027 | 0.0 | 0.000 |
| Logistic Regression (pixels) | raw | 1.000 | 1.000 | 0.699 | 0.677 | +0.301 | 7.3 | 0.005 |
| Random Forest (pixels) | raw | 0.998 | 0.998 | 0.813 | 0.798 | +0.185 | 39.6 | 0.043 |
| MLP 256-128 (pixels) | raw | 1.000 | 1.000 | 0.783 | 0.761 | +0.217 | 7.0 | 0.009 |
| Linear SVM (HOG) | hog | 1.000 | 1.000 | 0.891 | 0.886 | +0.108 | 310.7 | 0.015 |
| MLP 256-128 (HOG) | hog | 1.000 | 1.000 | 0.906 | 0.898 | +0.094 | 9.2 | 0.013 |
