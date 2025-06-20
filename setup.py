# setup.py
from setuptools import setup, find_packages

setup(
    name="chiprag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "networkx",
        "matplotlib",
        "scikit-learn",
        "torch",
        "transformers",
        "spacy",
        "nltk",
        "gensim",
        "faiss-cpu",
        "sentence-transformers",
        "pytest",
        "pytest-cov"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="ChipRAG - 芯片设计RAG系统",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chip-rag",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
