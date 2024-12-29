# Safe DNA Storage

A Python library for encoding data into DNA sequences while ensuring biological safety. The system generates DNA sequences that don't encode for potentially harmful proteins.

## Features
- Text and image encoding into safe DNA sequences
- Multi-threaded processing for large data
- Protein toxicity prediction using ToxinPred3.0: https://github.com/raghavagps/toxinpred3
- Error correction using Reed-Solomon coding
- Chunked processing for large data

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Text Encoding
```python
from dna_storage import ToxinPredictor, DNAAnalyzer, find_safe_encoding_chunked

# Initialize components
predictor = ToxinPredictor(model_path="model/toxinpred3.0_model.pkl")
analyzer = DNAAnalyzer(predictor)

# Encode text
safe_dnas, results = find_safe_encoding_chunked(
    text="Hello World",
    analyzer=analyzer,
    chunk_size=15,  # Characters per chunk
    initial_variants=20,  # Initial number of DNA variants to try
    max_variants=100  # Maximum variants if initial attempt fails
)
```

### Image Processing
```python
from dna_storage import process_image

safe_dnas, results = process_image(
    image_path="input.jpg",
    output_path="output.jpg",
    dna_output_path="dna_sequences.json",
    analyzer=analyzer,
    chunk_size=56
)
```

## How It Works

1. **Input Processing**: Text/images are split into manageable chunks
2. **DNA Encoding**: Each chunk is encoded into multiple DNA sequence variants
3. **Safety Check**: Each variant is analyzed for potential toxic proteins
4. **Selection**: The safest variant that successfully encodes the data is selected
5. **Error Correction**: Reed-Solomon coding ensures data integrity

## Parameters

- `chunk_size`: Number of characters per DNA sequence chunk
- `initial_variants`: Starting number of DNA variants to generate
- `max_variants`: Maximum variants to try if initial attempts fail
- `ecc_symbols`: Number of error correction symbols (You need to use the same value when decoding)
- `best_of`: Number of attempts for each variant generation


## Main contributors:

* [Diane Letourneur](https://www.linkedin.com/in/diane-letourneur-396a7a25a)
* [Quentin Feuillade--Montixi](https://www.linkedin.com/in/quentin101010/)