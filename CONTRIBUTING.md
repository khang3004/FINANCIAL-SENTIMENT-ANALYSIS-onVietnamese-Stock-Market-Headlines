# Contributing to Vietnamese Financial Sentiment Analysis

Thank you for your interest in contributing! This document provides guidelines for contributing.

## 🎯 How to Contribute

### 1. Reporting Bugs

- Use GitHub Issues
- Include detailed description
- Provide reproduction steps
- Add expected vs actual behavior

### 2. Feature Requests

- Open a GitHub Issue with "Feature Request" label
- Describe the feature and its benefits
- Explain use cases

### 3. Code Contributions

#### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/financial-sentiment-analysis.git
cd financial-sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

#### Code Style

- Follow PEP 8 guidelines
- Use Black for formatting: `black src/ tests/`
- Add type hints where possible
- Write docstrings (Google style)

#### Testing Requirements

- Write unit tests for new features
- Maintain >90% code coverage
- Run all tests before submitting: `pytest tests/ -v --cov=src`

#### Pull Request Process

1. Create feature branch: `git checkout -b feature/amazing-feature`
2. Make changes and commit: `git commit -m 'Add amazing feature'`
3. Push to branch: `git push origin feature/amazing-feature`
4. Open Pull Request
5. Ensure all CI checks pass
6. Request review from maintainers

## 📋 Code Review Guidelines

### Before Submitting PR

- [ ] Code follows style guidelines
- [ ] Tests are written and passing
- [ ] Documentation is updated
- [ ] No sensitive data committed
- [ ] Commit messages are clear

### Review Checklist

- Code quality and readability
- Test coverage
- Documentation completeness
- Performance implications
- Security considerations

## 🚀 Release Process

1. Version bump in `setup.py`
2. Update CHANGELOG.md
3. Create release tag
4. Publish to PyPI (if applicable)

## 💬 Communication

- Use GitHub Issues for technical discussions
- Be respectful and constructive
- Help others learn

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.
