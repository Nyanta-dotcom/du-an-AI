NGŨ ĐẠI HÀO KIỆT
                                             
                                       .                                            
                                      .. ..                                         
                                       .......                                      
                                         .--  ..                                    
                                .---..     ........                                 
                          --...--+--++++-++-.-.....  .                              
                      .-  .-++++++++++++++++++-.................                    
                    -   -++++++++++++###+++++++++-......  ...    ......             
                  ..   -+++++++++++++#####++++++++----........    ....              
                 -    ++++++++++++#############+++++----..   ............           
                ..  -++++++++++###++++++++++###-          ....    .----...          
                +  +++++++######++++++++++++++-.            ... .-....-...          
                + -++###+++##+++#++++++++++++++.... ..    .... --------  ..         
                +--+######++++++++++++++++++++++---.  .....   --------+-..-.        
               +++ +##+++++++++++++++++-----++++ ...---  .- -+++++++-+---...        
              +++++##++--+++++++++++++++++++++-.       ..- -++++++++++---....       
             -++++#+#---++#####+++++++++####+++. .     .. -++++++++++++--.-...      
             -++++#++.---++++++++++++++#++++++---.....    -+++++++++++++--.-  .     
             ++++##++#--+++++#++++++++##+##++++++++----.-+-  -++++++++++++.-.--.    
            .+######---..-+++++++++++++++#+++--++###+-+++++++-  .+++++++++...---    
             +######--------++++++--+++++++++--++####+-+++++++++-   ++++++-..----   
             +######--------+++-----++++++++++-++#####++++++++++++++   -+++..-----  
             +######+-----------------++++++++--+######  -+++++++++++++.  --.------ 
             .+######----+++++++++++++++++++++++######+   .-++++++++++++++.   -----.
               +#+###---++++++-+++++++++++++++++#####.   ....-++++++++++++++-+++----
                 - #..-+++++++-+++++++++++++++++-.    +++++-----+++++++++++++++++---
                 -  #---+++++##########+++++++#++++#++#++----------+++++++++++++++--
               .-++-+++++++++++++++++++++++++#########+#+++++++----.-++++++++++++-+-
        ..----+-+++++##++++++++++++##+++++++###############++++++++---+++++++++++++-
     ..----++++-+++++##+++++++++++++++++++#+##################++++++---+++++++++++++
# Real-time Sign Language Alphabet Recognition

AI course final project: a web app that recognises the ASL alphabet from a webcam in real time ("Duolingo for sign language").
The browser app (`web/`) uses MediaPipe Hands + a landmark MLP; the image-based CNN on Sign Language MNIST is the comparison model.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Download the datasets into `data/raw/` as described in [data/README.md](data/README.md).

## Run (from the project root)

```bash
# Image pipeline (Sign Language MNIST)
python -m src.data                # quality checks + preprocessing -> data/processed/sign_mnist.npz
python -m src.eda                 # data figures -> reports/figures/
python -m src.train_baselines     # week 3 baselines -> reports/baseline_metrics.{json,md}
python -m src.train_cnn_numpy     # week 4 CNN, 4 configs (~30 min CPU) -> models/, reports/week4/
python -m src.week4_eval          # comparison + error analysis -> reports/week4/

# Landmark pipeline (web app model)
python -m src.train_landmark_mlp  # -> models/landmark_mlp_bundle.json, reports/landmark/metrics.json
python web/build_app.py           # embed the model into web/index.html

# Web app
web\run_app.bat                   # or: cd web && python -m http.server 8000 -> http://localhost:8000
```

All randomness is seeded (`SEED = 42` in `src/config.py`).

## Structure

```
src/                   Python: data, features, models, training, evaluation
  config.py            paths, constants, label mapping (24 letters, no J/Z)
  data.py              load -> quality report -> clean -> scale -> group split
  features.py          raw pixels, HOG features
  augment.py           rotation / shift / zoom / brightness / contrast augmentation
  eda.py               data understanding figures
  train_baselines.py   week 3 baselines + metrics + confusion matrix
  cnn_numpy.py         CNN implemented in NumPy
  train_cnn_numpy.py   week 4 CNN training (4 hyper-parameter configs)
  week4_eval.py        baseline vs CNN comparison, error analysis
  train_landmark_mlp.py  MLP on MediaPipe hand landmarks (web app model)
models/                trained weights (CNN .npz, landmark MLP bundle)
web/                   front-end: static HTML/JS app, runs fully in the browser
reports/               outputs generated by the code (metrics, figures)
docs/                  written documents
  course/              course brief and weekly requirements
  reports/             weekly reports
  assets/              reference images
data/                  datasets (not committed, see data/README.md)
```

## Notes

- Labels 9 (J) and 25 (Z) do not exist in the dataset because those letters need motion; the models cover 24 static letters.
- The official test set is used only for final reporting, never for model selection.
