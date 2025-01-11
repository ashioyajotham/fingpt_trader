# Contributing to FinGPT Trader

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/ashioyajotham/fingpt-trader.git
   cd fingpt-trader
   ```

2. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   pre-commit install
   ```

## Development Workflow

### 1. Branching Strategy

- `main`: Production-ready code
- `develop`: Development branch
- Feature branches: `feature/your-feature`
- Bugfix branches: `fix/issue-description`

```bash
# Create feature branch
git checkout -b feature/your-feature develop

# Create bugfix branch
git checkout -b fix/issue-description develop
```

### 2. Code Style

We follow PEP 8 with these modifications:

```python
# Maximum line length
max_line_length = 88  # Black formatter default

# Import ordering
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Class definitions
class YourClass:
    """Docstring following Google style."""
    
    def __init__(self, param: str):
        """
        Args:
            param: Description
        """
        self.param = param
```

### 3. Testing

All code must include tests:

```python
# test_your_feature.py
import pytest
from your_module import YourClass

def test_your_feature():
    obj = YourClass("test")
    assert obj.method() == expected_result
    
@pytest.mark.parametrize("input,expected", [
    ("case1", result1),
    ("case2", result2)
])
def test_parametrized(input, expected):
    assert function(input) == expected
```

Run tests:
```bash
pytest tests/
pytest tests/your_specific_test.py -v
```

### 4. Documentation

1. **Docstrings**: Google style
   ```python
   def function(param1: str, param2: int) -> Dict[str, Any]:
       """Short description.
       
       Longer description if needed.
       
       Args:
           param1: Description of param1
           param2: Description of param2
           
       Returns:
           Dict containing processed results
           
       Raises:
           ValueError: If param1 is invalid
       """
   ```

2. **README Updates**: Keep module READMEs updated
3. **API Documentation**: Update API docs for public interfaces

### 5. Pull Request Process

1. **Before Submitting**
   - Run all tests
   - Update documentation
   - Add changelog entry
   - Rebase on latest develop

2. **PR Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   
   ## Testing
   - [ ] Added unit tests
   - [ ] Updated existing tests
   
   ## Documentation
   - [ ] Updated docstrings
   - [ ] Updated README
   - [ ] Updated API docs
   ```

3. **Review Process**
   - Two approvals required
   - All tests must pass
   - Documentation must be complete

### 6. Performance Considerations

1. **Profiling**
   ```python
   import cProfile
   import pstats
   
   def profile_code():
       profiler = cProfile.Profile()
       profiler.enable()
       # Your code here
       profiler.disable()
       stats = pstats.Stats(profiler).sort_stats('cumtime')
       stats.print_stats()
   ```

2. **Benchmarks**
   ```python
   import timeit
   
   def benchmark_function():
       setup = "from your_module import your_function"
       stmt = "your_function(test_data)"
       return timeit.timeit(stmt, setup, number=1000)
   ```

## Project Structure

When adding new features, follow the existing structure:

```
models/
├── new_feature/
│   ├── __init__.py
│   ├── core.py
│   ├── utils.py
│   └── README.md  # Feature documentation
tests/
└── new_feature/
    ├── __init__.py
    ├── test_core.py
    └── test_utils.py
```

## Common Tasks

### Adding a New Model

1. Create model structure:
   ```bash
   mkdir models/your_model
   touch models/your_model/{__init__,core,utils}.py
   ```

2. Create tests:
   ```bash
   mkdir tests/your_model
   touch tests/your_model/test_{core,utils}.py
   ```

3. Update documentation:
   - Add model README
   - Update API docs
   - Add examples

### Fixing Bugs

1. Create issue if none exists
2. Create fix branch
3. Add regression test
4. Update changelog
5. Submit PR

## Getting Help

- Check [Discussions](https://github.com/ashioyajotham/fingpt-trader/discussions)
- Tag maintainers in issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License. Read more in [LICENSE](LICENSE).