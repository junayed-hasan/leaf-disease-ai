# Contributing to TomatoLeaf-AI

Thank you for your interest in contributing to TomatoLeaf-AI! This document provides guidelines and information for contributors.

## 🌟 How to Contribute

### Reporting Issues

1. **Check existing issues** first to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Operating system and version
   - Python version
   - PyTorch version
   - Steps to reproduce the issue
   - Expected vs actual behavior
   - Error messages and stack traces

### Suggesting Enhancements

1. **Check existing feature requests** to avoid duplicates
2. **Clearly describe the enhancement**:
   - Use case and motivation
   - Proposed solution
   - Alternative solutions considered
   - Impact on existing functionality

### Code Contributions

#### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/lightweight-tomato.git
   cd lightweight-tomato
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_balancing.txt
   pip install -e ".[dev]"
   ```

4. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Code Style and Standards

- **Python Style**: Follow PEP 8 guidelines
- **Code Formatting**: Use `black` for code formatting
- **Import Sorting**: Use `isort` for import organization
- **Linting**: Use `flake8` for code linting
- **Type Hints**: Add type hints where appropriate

```bash
# Format code
black src/ scripts/

# Sort imports
isort src/ scripts/

# Lint code
flake8 src/ scripts/
```

#### Testing

- **Write tests** for new functionality
- **Ensure existing tests pass**
- **Maintain test coverage** above 80%

```bash
# Run tests
pytest tests/

# Run tests with coverage
pytest --cov=src tests/
```

#### Documentation

- **Update docstrings** for new functions and classes
- **Update README.md** if adding new features
- **Add examples** for new functionality
- **Update relevant documentation** in `docs/`

#### Commit Guidelines

Follow conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(distillation): add temperature scaling for knowledge distillation
fix(quantization): resolve INT8 quantization accuracy issues
docs(readme): update installation instructions
```

## 🔬 Research Contributions

### Dataset Contributions

- **New datasets**: Follow the existing dataset structure
- **Data preprocessing**: Ensure compatibility with existing pipeline
- **Documentation**: Provide detailed dataset description

### Model Contributions

- **New architectures**: Add to `src/models/`
- **Performance benchmarks**: Include comparative results
- **Mobile compatibility**: Ensure edge deployment feasibility

### Algorithm Contributions

- **Knowledge distillation**: Extend existing KD framework
- **Quantization methods**: Add new quantization techniques
- **Ensemble methods**: Improve ensemble strategies

## 📋 Pull Request Process

1. **Ensure your code follows** the style guidelines
2. **Update documentation** as needed
3. **Add or update tests** for your changes
4. **Ensure all tests pass**
5. **Update the changelog** if applicable
6. **Create a detailed pull request**:
   - Clear title and description
   - Reference related issues
   - Include screenshots for UI changes
   - List breaking changes

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## 🏆 Recognition

Contributors will be recognized in:
- **README.md** acknowledgments
- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions

## 📞 Getting Help

- **GitHub Discussions**: For questions and discussions
- **Issues**: For bug reports and feature requests
- **Email**: Contact maintainers directly for sensitive issues

## 🎯 Priority Areas

We especially welcome contributions in:

1. **Mobile Optimization**
   - Model compression techniques
   - Inference optimization
   - Mobile app improvements

2. **Cross-Domain Generalization**
   - Domain adaptation methods
   - Transfer learning techniques
   - Robustness improvements

3. **Explainable AI**
   - New interpretability methods
   - Visualization improvements
   - Biological validation

4. **Dataset Expansion**
   - New disease classes
   - Field condition datasets
   - Multi-language annotations

## 📜 Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Harassment of any kind
- Discriminatory language or actions
- Personal attacks or insults
- Publishing private information without permission
- Other conduct inappropriate in a professional setting

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project maintainers. All complaints will be reviewed and investigated promptly and fairly.

## 🙏 Thank You

Your contributions help advance agricultural AI research and support farmers worldwide. Every contribution, no matter how small, makes a difference!

---

For questions about contributing, please open an issue or contact the maintainers directly. 