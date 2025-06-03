---
weight: 100
date: "2023-05-03T22:37:22+01:00"
draft: false
author: "Guilherme Oliveira"
title: "Quickstart"
icon: "rocket_launch"
toc: true
description: "A quickstart guide to downloading and installing pymocd library"
publishdate: "2025-02-01T22:37:22+01:00"
tags: ["Beginners"]
---

## Requirements

---
| Requirement          | Version        | Description                                        |
|----------------------|----------------|----------------------------------------------------|
| **Python**           | ≥ 3.9          | Required to run the library                        |
| **pip**              | ≥ 24.3.1       | Python package installer                            |
| **Python venv**      | Built-in       | Recommended for isolated environments (optional)    |

---

## Install pymocd

{{< tabs tabTotal="3">}}
{{% tab tabName="Linux" %}}

Most Linux distributions come with **python** and **pip** pre-installed. Create a virtual environment and install the library:

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install pymocd
```

{{% /tab %}}

{{% tab tabName="Homebrew (macOS)" %}}

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install pymocd
```

{{% /tab %}}

{{% tab tabName="Windows" %}}

On Windows, use PowerShell or Command Prompt:

```powershell
> python -m venv .venv
> .\.venv\Scripts\Activate.ps1
> pip install pymocd
```

If using Command Prompt (cmd.exe):

```cmd
> python -m venv .venv
> .\.venv\Scripts\activate.bat
> pip install pymocd
```

{{% /tab %}}
{{< /tabs >}}

{{< alert text="We recommend installing `networkx` or `igraph` and `matplotlib` alongside pymocd for full functionality!" />}}

```bash
$ pip install networkx matplotlib
```

## Compiling from Source

For the latest version, you can compile pymocd from the GitHub repository's main branch using **maturin**.

First, clone the repository:

```bash
$ git clone https://github.com/oliveira-sh/pymocd/ && cd pymocd
```

Install maturin to build the library inside a virtual environment:

```bash
$ pip install maturin
```

Then compile the project:

```bash
$ maturin build --release
```