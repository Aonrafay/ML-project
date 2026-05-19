# Contributing to SocialPulse AI

Thank you for your interest in contributing to SocialPulse AI! This guide outlines the process and standards for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

- Be respectful and constructive in all interactions
- Focus on what is best for the community and the project
- Gracefully accept constructive criticism

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/socialpulse-ai.git
   cd "ML project"
   ```
3. Create a branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB (local or Atlas)
- Redis
- Docker (optional, for infrastructure services)

### Backend Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Infrastructure

```bash
docker-compose up -d
```

### Environment Configuration

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Project Structure

```
socialpulse-ai/
├── backend/               # FastAPI backend
│   └── app/
│       ├── api/            # API route handlers
│       ├── core/           # Config, database, security
│       ├── models/         # Data models
│       ├── schemas/        # Pydantic schemas
│       ├── services/       # Business logic services
│       └── utils/          # Helper utilities
├── data_collection/        # Reddit & Twitter collectors
├── data_processing/        # Text preprocessing & pipeline
├── ml_models/              # ML model training & prediction
│   ├── fake_news/          # RoBERTa fake news detector
│   ├── sentiment/          # DistilBERT + VADER sentiment
│   └── topic_modeling/     # BERTopic topic clustering
├── fact_check/             # Claim verification APIs
├── frontend/               # Next.js + React dashboard
├── tests/                  # Test suites
├── deployment/             # Docker & production configs
└── docs/                   # Documentation
```

## Coding Standards

### Python

- Follow **PEP 8** style guidelines
- Use **type hints** for all function signatures
- Write **docstrings** for public functions and classes (Google style)
- Maximum line length: **120 characters**
- Use **async/await** for all database operations and I/O-bound tasks
- Organize imports: standard library → third-party → local

Example:

```python
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Query

from backend.app.core.database import raw_posts_collection


async def fetch_posts(
    platform: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """Fetch raw posts from the database with optional filters.

    Args:
        platform: Filter by platform (reddit or x).
        limit: Maximum number of posts to return.

    Returns:
        List of post dictionaries.
    """
    query = {}
    if platform:
        query["platform"] = platform
    cursor = raw_posts_collection.find(query).limit(limit)
    return await cursor.to_list(length=limit)
```

### TypeScript / React

- Use **TypeScript** for all new files
- Follow the **Next.js App Router** conventions
- Use **functional components** with hooks
- Prefer **named exports** over default exports
- Use **CSS Modules** or **Tailwind CSS** for styling

### General

- Keep functions **small and focused** (max ~50 lines)
- Avoid **deep nesting** (max 3 levels)
- Use **descriptive variable names** — no single-letter names except in loops
- Remove **unused code** and commented-out blocks
- Handle **errors gracefully** with proper logging

## Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Code style changes (formatting, etc.) |
| `refactor` | Code refactoring without feature changes |
| `test` | Adding or updating tests |
| `chore` | Build process or auxiliary tool changes |
| `perf` | Performance improvements |

### Examples

```
feat(api): add export endpoint with CSV support
fix(sentiment): correct VADER threshold for neutral classification
docs(readme): update quick start guide with Docker instructions
test(preprocessing): add unit tests for text cleaning functions
```

## Pull Request Process

1. **Update documentation** if your changes affect behavior
2. **Add tests** for any new functionality
3. **Ensure all tests pass** before submitting:
   ```bash
   python -m pytest tests/ -v
   cd frontend && npm test
   ```
4. **Keep PRs focused** — one feature or fix per PR
5. **Write a clear PR description** explaining:
   - What changes were made
   - Why they were made
   - How to test them
6. **Request review** from a maintainer
7. **Address review feedback** promptly and push fixes

### PR Checklist

- [ ] Code follows project coding standards
- [ ] Tests added/updated for new functionality
- [ ] All existing tests pass
- [ ] Documentation updated (README, docs/, inline comments)
- [ ] No breaking changes without migration guide
- [ ] Environment variables documented in `.env.example`

## Testing

### Running Tests

```bash
# Backend tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=backend --cov=ml_models --cov-report=html

# Frontend tests
cd frontend && npm test
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files: `test_<module>.py`
- Use **pytest** fixtures for database setup/teardown
- Test both **success and error** cases
- Aim for **80%+ coverage** on critical paths

Example:

```python
import pytest
from data_processing.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


def test_clean_text_removes_urls(preprocessor):
    text = "Check this out! https://example.com/news"
    result = preprocessor.clean_text(text)
    assert "https://example.com" not in result
    assert "Check this out!" in result


def test_detect_language_english(preprocessor):
    text = "This is clearly an English sentence."
    result = preprocessor.detect_language(text)
    assert result == "en"
```

## Reporting Issues

When reporting bugs or requesting features:

1. **Use GitHub Issues** with the appropriate template
2. **Provide clear reproduction steps** for bugs
3. **Include environment details** (Python version, OS, etc.)
4. **Attach logs or screenshots** if applicable
5. **Tag issues appropriately** (bug, feature, enhancement, etc.)

### Bug Report Template

```
**Description**: Brief description of the issue

**Steps to Reproduce**:
1. Step one
2. Step two
3. Step three

**Expected Behavior**: What should happen

**Actual Behavior**: What actually happens

**Environment**:
- Python version:
- OS:
- MongoDB version:
- Any relevant config from .env
```

## Questions?

If you have questions about contributing, feel free to open a GitHub Issue with the `question` tag.

Thank you for helping make SocialPulse AI better!