# OmniGen TensorFlow

TensorFlow implementation of OmniGen multi-modal generative AI model.

## Installation

```bash
git clone https://github.com/H-devigner/OmniGen-tensorflow.git
cd OmniGen-tensorflow
pip install -e .
```

## Requirements

- Python >= 3.8
- TensorFlow >= 2.13.0
- transformers
- pillow
- numpy
- huggingface-hub

## Usage

```python
from omnigen_tf import OmniGenTFProcessor

# Initialize processor
processor = OmniGenTFProcessor.from_pretrained("path/to/model")

# Process text
text_inputs = processor.process_text("Generate an image of a cat")

# Process image
image_inputs = processor.process_image("path/to/image.jpg")

# Process multi-modal input
multi_modal = processor.process_multi_modal_prompt(
    "Generate an image similar to <|image_1|> but with a different style",
    ["path/to/reference.jpg"]
)
```

## Features

- Multi-modal input processing (text + images)
- Batch processing support
- Flexible image size handling
- Comprehensive error handling
- Type hints throughout codebase

## License

MIT License