# Setup configuration for Vietnamese Financial Sentiment Analysis

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vietnamese-financial-sentiment",
    version="1.0.0",
    author="Vietnamese Financial Sentiment Analysis Team",
    description="Production-ready sentiment analysis for Vietnamese financial markets using state-of-the-art NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/financial-sentiment-analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "pydantic>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-sentiment=scripts.train:main",
        ],
    },
)
