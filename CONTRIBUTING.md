# Contributing to PHI

Thank you for your interest in contributing to the PHI (Golden Ratio AI Training Framework) project! This document provides guidelines for contributing to this research and educational project.

## 📋 Code of Conduct

By participating in this project, you agree to abide by our code of conduct:
- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a welcoming environment for all contributors
- Respect the non-commercial nature of this project

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic understanding of machine learning concepts
- Familiarity with the golden ratio mathematical principles

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/PHI.git
   cd PHI
   ```
3. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run tests to ensure everything works:
   ```bash
   pytest tests/
   ```

## 🔬 Research Areas

We welcome contributions in these areas:

### Core PHI Research
- Mathematical analysis of golden ratio applications in ML
- New PHI-based optimization algorithms
- Theoretical foundations and proofs

### Implementation Improvements
- Performance optimizations
- New training strategies
- Dashboard enhancements
- Documentation improvements

### Educational Content
- Tutorials and examples
- Mathematical explanations
- Visualization improvements
- Case studies

## 📝 Contribution Types

### 🐛 Bug Reports
- Use the issue template
- Include reproduction steps
- Provide system information
- Include relevant logs/screenshots

### 💡 Feature Requests
- Describe the research motivation
- Explain the expected behavior
- Consider educational value
- Discuss implementation approach

### 🔧 Code Contributions
- Follow the coding standards below
- Include tests for new features
- Update documentation
- Ensure backward compatibility

## 📊 Coding Standards

### Python Style
- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for all public functions
- Maximum line length: 100 characters

### Documentation
- Update README.md for major features
- Include inline comments for complex algorithms
- Provide examples for new functionality
- Document mathematical foundations

### Testing
- Write unit tests for new functions
- Include integration tests for major features
- Ensure all tests pass before submitting
- Aim for >80% code coverage

## 🔄 Development Workflow

1. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards

3. **Test your changes**:
   ```bash
   pytest tests/
   python -m phi.cli --help  # Test CLI
   streamlit run dashboard/streamlit_app.py  # Test dashboard
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add golden ratio batch scheduler"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** with:
   - Clear description of changes
   - Link to related issues
   - Screenshots for UI changes
   - Test results

## 🧪 Research Validation

For research contributions:
- Include mathematical proofs or references
- Provide experimental validation
- Compare against baseline methods
- Document limitations and assumptions

## 📚 Documentation Guidelines

- Use clear, educational language
- Include mathematical notation where appropriate
- Provide practical examples
- Link to relevant research papers
- Update the changelog for significant changes

## ⚖️ License Compliance

- All contributions must be compatible with CC BY-NC 4.0
- Do not include commercial code or dependencies
- Respect third-party licenses
- Clearly document any external research references

## 🤝 Community

- Join discussions in GitHub Issues
- Share research findings and insights
- Help other contributors
- Participate in code reviews

## 📞 Getting Help

- Check existing issues and documentation
- Ask questions in GitHub Discussions
- Review the PHI Training Guide
- Contact maintainers for research collaboration

## 🎯 Roadmap

Current focus areas:
- Advanced PHI optimization algorithms
- Multi-modal training support
- Distributed training capabilities
- Educational content expansion

Thank you for contributing to PHI research! 🔬✨
