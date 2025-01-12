from setuptools import find_packages, setup
import re

def get_requirements():
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    
    processed_requirements = []
    for req in requirements:
        req = req.strip()
        # Skip comments and empty lines
        if not req or req.startswith('#'):
            continue
            
        # Convert exact versions to minimum versions
        # Handle both == and === version specifiers
        req = re.sub(r'={2,3}', '>=', req)
        
        # Remove any trailing comments after version specifier
        req = req.split('#')[0].strip()
        
        processed_requirements.append(req)
    
    return processed_requirements

setup(
    name="fingpt-trader",
    version="0.1.0",
    description="AI-powered trading system using FinGPT",
    author="Victor Jotham Ashioya",
    author_email="ashioyajotham@icloud.com",
    packages=find_packages(),
    install_requires=get_requirements(),
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