# Animal Sound Classifier
> Course Project | SDSC2102 - Statistical Methods and Data Analysis
- Machine Leraning for animal souind classification using logistic regression and decision tree model
- Audio dataset from ESC-50.

# Setup

> This setup instruction is follow my own implementation for preparing the environment for this project.

Step 1. Clone or download this repository and set it as the working directory.
Open the working directory with terminal and run: `git clone https://github.com/YeungYuiii/Animal-Sound-Cassification-SDSC2102-.git`

Step 2. Preparing ESC-50 raw audio dataset with DVC.
Change directory to the project directory: `cd Animal-Sound-Cassification-SDSC2102-`
Pull only the audio file from ESC-50's repositories: 
`git fetch https://github.com/karolpiczak/ESC-50.git`
`git checkout FETCH_HEAD -- audio`

Step 3. Create a virtual environment and install the dependencies.
`python3 -m venv sdsc2102-asc`
`sdsc2102-asc\Scripts\activate.bat`
`pip install -r requirements.txt`

