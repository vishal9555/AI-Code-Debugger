# AI-Based Code Debugger

This project implements an AI-Based Code Debugger that utilizes machine learning models to identify, classify, and suggest fixes for errors in source code. The system is designed to analyze code snippets, detect syntax and logical errors, and recommend corrections using deep learning and natural language processing techniques.

## Table of Contents
- Project Structure
- Detailed Code Explanation
  1. Data Collection and Preprocessing
  2. Model Architecture
  3. Training the Debugger
  4. Interactive Debugging
- Dependencies
- Getting Started
- Running the Project
- Additional Information
- License
- Contributing

## Project Structure
```
AI-Code-Debugger/
├── app.py                   # Main script: loads data, preprocesses code, trains the AI model, and performs debugging.
├── notebook.ipynb           # Jupyter Notebook with an interactive walkthrough of data preprocessing, model training, and inference.
├── datasets/
│   ├── code_samples.json    # JSON dataset containing labeled code snippets with errors and corrections.
│   └── additional_data.csv  # Supplementary dataset with real-world programming mistakes.
├── models/
│   ├── model.pth            # Pretrained model weights.
│   └── model_arch.py        # Defines the deep learning model architecture.
├── .gitignore               # Git ignore rules.
└── README.md                # Main documentation file.
```

## Detailed Code Explanation

### 1. Data Collection and Preprocessing
- **Dataset:** The system uses labeled datasets containing buggy and corrected code snippets.
- **Parsing and Tokenization:** Code samples are tokenized into AST (Abstract Syntax Tree) representations using libraries like `tree-sitter`.
- **Feature Extraction:** Extracts syntactic and semantic features from the code, including indentation patterns, variable usage, and common error patterns.
- **Normalization:** All code samples are converted into a standardized format to improve generalization.

### 2. Model Architecture

#### `DebuggerModel` Class
- **Embedding Layer:** Converts tokenized code sequences into vector representations.
- **LSTM/Transformer Layers:** Captures long-range dependencies in code structure.
- **Classification Head:** Predicts the type of error (e.g., syntax, logical, runtime).
- **Correction Module:** Suggests possible fixes using sequence-to-sequence learning.

### 3. Training the Debugger

- **Loss Function:** Combines cross-entropy loss for error classification and sequence loss for correction generation.
- **Optimization:** Uses Adam optimizer with learning rate scheduling.
- **Training Process:**
  - Trains on labeled datasets containing incorrect and corrected code.
  - Uses data augmentation techniques like variable renaming and indentation changes.
  - Periodic evaluation on a validation set.

### 4. Interactive Debugging

After training, the system allows users to input code snippets for analysis.

- **Error Detection:** Identifies issues in the input code and highlights potential errors.
- **Correction Suggestions:** Suggests possible fixes ranked by confidence score.
- **Code Completion:** Provides auto-suggestions based on learned patterns.

## Dependencies
The project requires the following Python libraries:

- `torch` (for deep learning model training)
- `transformers` (for NLP-based models)
- `tree-sitter` (for parsing code syntax)
- `pandas` (for handling datasets)
- `numpy` (for numerical operations)

To install the required dependencies:
```bash
pip install torch transformers tree-sitter pandas numpy
```

## Getting Started

### Prepare Your Dataset
Ensure that your dataset (`code_samples.json`) is correctly formatted and placed in the `datasets/` directory.

### Train the Model
Run the training script to train the debugger:
```bash
python app.py --train
```

### Interactive Debugging
After training, start an interactive debugging session:
```bash
python app.py --debug
```

The system will prompt for code input and return error classifications and corrections.

## Running the Project

Using the Command Line:
```bash
python app.py --debug
```

Using Jupyter Notebook:
Open `notebook.ipynb` in Jupyter Notebook or VS Code’s interactive window to walk through the steps of data processing, model training, and debugging.

## Additional Information

- **Error Categories:** The system supports detecting multiple types of errors, including syntax errors, logical errors, and runtime exceptions.
- **Model Customization:** Modify hyperparameters such as learning rate and model depth in `model_arch.py`.
- **Evaluation:** Performance is measured using accuracy, precision-recall, and BLEU score for code correction suggestions.

## License
This project is released under the MIT License.

## Contributing
Contributions are welcome! Please create a pull request with your improvements or bug fixes.




