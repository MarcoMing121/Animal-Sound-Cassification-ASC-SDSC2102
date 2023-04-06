# Animal Sound Classifier
> Course Group Project | SDSC2102 - Statistical Methods and Data Analysis <br />
> Group members: Max, Marco, Ivan, Noddy from City University of Hong Kong
- data-oriented audio analysis.
- data preparation and engineering for audio signal dataset.
- Machine Leraning for animal souind classification using logistic regression and decision tree model.
- Audio dataset from ESC-50.

# File Structure
Documentation for explaining our implementation method
```
1. Data Preparation doc v2.pdf
explain the method for animal sound extraction and the features extracted

2. 2102 model doc.pdf
explain modeling tasks by Marco.
```
Code for animal sound extraction and features extraction
```
1. AnimalDetect.py
module for animal sound extraction and features extraction

2. data-visualization.ipynb
audio signal and spectrogram visualiztion

3. detection-clips.ipynb
prototype of animal sound clipping

4. detection-cut.ipynb
prototype of second cutting with UCL and LCL

5. extraction.py
code for extracting features and save them into csv file

6. animal_features.csv
extracted features for all animals
```
Code for animal sound classification modeling
```
1. asc_ML.ipynb
code for following modeling tasks by Marco
 - source-to-target mapping
 - cross validation
 - logistic regression training
 - decision tree training
 - model evaluation
```

# Setup

> This setup instruction is follow my own implementation for preparing the environment for this project.

Step 1. Clone or download this repository and set it as the working directory.
- Open the working directory with terminal and run: 
```
git clone https://github.com/YeungYuiii/Animal-Sound-Cassification-SDSC2102-.git
```

Step 2. Preparing ESC-50 raw audio dataset with DVC.
- Change directory to the project directory: 
```
cd Animal-Sound-Cassification-ASC-SDSC2102
```
- Pull only the audio and meta from ESC-50's repositories: 
```
git fetch https://github.com/karolpiczak/ESC-50.git
git checkout FETCH_HEAD -- audio
git checkout FETCH_HEAD -- meta
```

Step 3. Create a virtual environment and install the dependencies.
```
python3 -m venv sdsc2102-asc
sdsc2102-asc\Scripts\activate.bat
pip install -r requirements.txt
```
