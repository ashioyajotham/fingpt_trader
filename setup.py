from setuptools import find_packages, setup

setup(
    name="fingpt-trader",
    version="0.1.0",
    description="AI-powered trading system using FinGPT",
    author="Victor Jotham Ashioya",
    author_email="ashioyajotham@icloud.com",
    packages=find_packages(),
    install_requires=[
        # ML/DL
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        # API/Data
        "aiohttp>=3.8.0",
        "pyyaml>=6.0",
        "alpha_vantage>=2.3.1",
        # Learning Components
        "stable-baselines3>=2.0.0",  # RL algorithms
        "optuna>=3.2.0",  # Hyperparameter optimization
        "mlflow>=2.7.0",  # Experiment tracking
        "ray[tune]>=2.6.0",  # Distributed training
        "backtrader>=1.9.76",  # Backtesting engine
        "gymnasium>=0.29.0",  # RL environments
        "wandb>=0.15.0",  # Weights & Biases tracking
        "pytorch-lightning>=2.0.0",  # Training framework
        # Storage & Versioning
        "dvc>=3.21.0",  # Model versioning
        "boto3>=1.28.0",  # Cloud storage
    ],
    extras_require={"dev": ["pytest", "black", "mypy", "flake8", "pytest-asyncio"]},
    entry_points={
        "console_scripts": [
            "fingpt-trade=scripts.live_trade:main",
            "fingpt-backtest=scripts.backtest:main",
            "fingpt-analyze=scripts.analyze:main",
        ]
    },
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)
