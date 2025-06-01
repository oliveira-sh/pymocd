# PYMOCD Contributing

Thank you for your interest in improving **pymocd**! We welcome contributions of all kinds—bug reports, documentation fixes, new features, or anything else that makes pymocd better. 
To ensure a positive and productive environment, please review the guidelines below before getting started.


## 1. Code of Conduct

All contributors are expected to follow our [Code of Conduct](./CODE_OF_CONDUCT.md). By participating in this project, you agree to uphold its standards for respectful, inclusive, and constructive communication. If you encounter unacceptable behavior, please let the maintainers know by opening an issue or contacting us directly.


## 2. Licensing

pymocd is distributed under the [GPLv3 License](./LICENSE). By contributing code or documentation, you agree to license your contributions under the same terms. For details, refer to the `LICENSE` file in this repository.

## 3. Getting Started

### 3.1 Local Development Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/oliveira-sh/pymocd
   cd pymocd
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install build dependencies**
   We use [maturin](https://www.maturin.rs/) to build the Rust extensions and package everything for Python.

   ```bash
   pip install maturin
   ```

4. **Build and install pymocd**

   ```bash
   maturin develop --release
   ```

   This command compiles any native extensions and installs the package into your virtual environment.

5. **Verify installation**

   ```bash
   python -c "import pymocd; print(pymocd)"
   ```

   You should see something like this: `<module 'pymocd' from '/home/anon/Desktop/pymocd/.venv/lib/python3.13/site-packages/pymocd/__init__.py'>`.

## Reporting Issues

If you find bugs, unexpected behavior, or have suggestions for improvements, please check the [issue tracker on GitHub](https://github.com/oliveira-sh/pymocd/issues) and follow this workflow:

1. **Search existing issues** to see if someone else has already reported the same problem.
2. If you don’t find a relevant issue, click **“New Issue.”**
3. Choose an issue template:

   * **Bug report**: Describe what you expected to happen, what actually happened, and how to reproduce the problem. Include any error messages or logs.
   * **Feature request (enhancement)**: Explain the desired functionality, why it’s useful, and any ideas for implementation.
   * **Documentation**: If documentation is missing or unclear, outline what’s confusing and how you’d like it improved.

### Issue Labels

We use the following labels to triage and organize issues:

* **bug**: A confirmed bug in functionality or documentation.
* **enhancement**: A request for new features, improvements, or sections.
* **documentation**: Anything related to docs, tutorials, or examples.
* **discussion**: A broader conversation that may lead to enhancements or bug fixes.
* **E-mentor**: An experienced contributor has volunteered to help new contributors tackle this issue.

If you’re new to open source contributions, look for issues tagged **E-mentor**—someone will guide you through the process.

## Submitting Pull Requests

We encourage you to submit pull requests (PRs) for any improvements. To ensure a smooth review process, please follow these steps:

1. **Fork the `master` branch**
   Create a copy of the repository on your GitHub account and clone it locally.

2. **Create a topic branch**
   Use a descriptive name, e.g.:

   ```bash
   git checkout -b fix-typo-in-docs
   ```

3. **Make your changes**

   * Stick to the existing code style (PEP 8 for Python; follow any project-specific conventions for Rust/C, if applicable).
   * Write clear commit messages (use the imperative mood: “Fix typo,” “Add new command,” etc.).
   * If you’re adding new functionality, include or update tests in the `tests/` directory.
   * If you modify public APIs, update the documentation in `docs/` accordingly.

4. **Push to your fork and open a PR**

   ```bash
   git push origin fix-typo-in-docs
   ```

   Then go to the original repository on GitHub and open a pull request from your branch into `master`.

## Documentation

* Documentation source files live in the `docs/` directory.
* We use [Hugo](https://gohugo.io/) to build docs. To build locally:

  ```bash
  cd docs
  hugo server
  ```

## Coding Guidelines

To maintain consistency across the project, please follow these guidelines:

* **Python Style**:

  * Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for naming, indentation (4 spaces), and line length (max 88 characters, as per black defaults).
  * Use type hints where appropriate. If you introduce new public-facing functions or classes, include type annotations.
  * Write docstrings in the [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).

* **Rust / Native Code Style (if applicable)**:

  * Follow the [Rust style guidelines](https://doc.rust-lang.org/1.0.0/style/).
  * Keep C bindings minimal and well-documented.
  * Ensure any `unsafe` blocks are accompanied by a clear explanation of why they’re needed and how they uphold safety guarantees.

* **Commit Messages**:

  * Use the imperative mood: “Add feature,” “Fix bug,” “Update docs,” etc.
  * Include a short summary (≤50 characters) and a more detailed description in the body (wrapped at 72 characters).

## Communication and Support

* **Discussion**: If you’re not sure where to start or have general questions, start a new issue with the **discussion** label or join our \[community chat/Slack/Discord] (link if available).
* **Pull Request Reviews**: We aim to review most pull requests within a few days. If you haven’t heard back, feel free to politely ping maintainers in the PR thread.
* **E-mentor Program**: Look for issues tagged **E-mentor**—an experienced contributor can help you get started.

## Thank You

Your time and effort are what make pymocd better. Whether you’re fixing a typo, writing documentation, reporting a bug, or adding a major feature, your contribution is invaluable. We appreciate all forms of participation—so let’s build something great together!

---

*Last updated: June 1, 2025*
