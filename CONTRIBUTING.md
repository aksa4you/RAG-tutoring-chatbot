# Contributing to RAG-Based Tutoring Chatbot

Thanks for your interest in contributing to this project! This document outlines how to get set up and the guidelines to follow when contributing.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) based tutoring chatbot built with:
- **Frontend/App**: Streamlit
- **Orchestration**: LangChain
- **Vector Store**: FAISS
- **Embeddings**: sentence-transformers
- **LLM**: Gemini 1.5 Flash

## Getting Started

1. **Fork the repository** and clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (e.g., your Gemini API key) in a `.env` file:
   ```
   GOOGLE_API_KEY=your_key_here
   ```

4. **Run the app locally**:
   ```bash
   streamlit run app.py
   ```

## How to Contribute

### Reporting Issues
- Check existing issues before opening a new one.
- Include clear steps to reproduce, expected behavior, and actual behavior.
- Mention your Python version and OS if it's a setup/deployment issue.

### Suggesting Enhancements
- Open an issue describing the enhancement and why it would be useful.
- For larger changes, please discuss before starting work to avoid duplicate effort.

### Submitting Changes
1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes, keeping commits focused and well-described.
3. Test your changes locally before submitting (see Testing section below).
4. Push your branch and open a Pull Request against `main`.
5. Describe what your PR does and reference any related issues.

## Code Style

- Follow PEP 8 for Python code.
- Use descriptive variable and function names.
- Add docstrings to new functions and classes.
- Keep functions focused on a single responsibility (e.g., keep ingestion, retrieval, and generation logic separate as in `ingest.py`, `rag_chain.py`).

## Testing

- If you modify the retrieval or generation pipeline, please re-run the RAGAS evaluation (`ragas_eval.py`) and report any change in faithfulness or answer relevancy scores.
- Current baseline: faithfulness ~0.87, answer relevancy ~0.83. Flag any regression below these in your PR.

## Areas Open for Contribution

- Improving chunking strategy or experimenting with chunk sizes
- Adding support for additional document formats
- Improving UI/UX in the Streamlit interface
- Expanding evaluation metrics
- Deployment and CI/CD improvements

## Questions?

Feel free to open an issue with the `question` label if anything is unclear.
