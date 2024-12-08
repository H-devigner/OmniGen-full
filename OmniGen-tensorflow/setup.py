from setuptools import setup, find_packages

setup(
    name="omnigen-tf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.13.0",
        "tensorflow-hub>=0.14.0",
        "torch>=2.0.0",
        "transformers>=4.31.0",
        "pillow>=9.5.0",
        "numpy>=1.24.3",
        "tqdm>=4.65.0",
        "huggingface-hub>=0.16.4",
        "safetensors>=0.3.1",
        "accelerate>=0.21.0",
        "typing-extensions>=4.5.0",
        "matplotlib>=3.7.0"
    ],
    author="H-devigner",
    author_email="",
    description="TensorFlow implementation of OmniGen multi-modal generative AI model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/H-devigner/OmniGen-tensorflow",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
