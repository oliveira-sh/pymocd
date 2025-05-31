---
weight: 450
title: "Limiting Core Usage in pymocd"
description: "Optional feature for controlling multi-threading behavior."
icon: power
lead: "How to limit CPU usage when using pymocd."
date: 2022-11-27T07:04:15+00:00
lastmod: 2023-08-11T17:38:15+00:00
draft: false
images: []
---

By default, `pymocd` uses [`rayon`](https://docs.rs/rayon), a Rust-based data-parallelism library that automatically leverages all available logical CPU threads. This is ideal for maximizing performance—but not always convenient when you need to run other tasks in parallel or are benchmarking performance under constrained conditions.

To limit CPU usage, you can set the number of threads used by `pymocd` with:

```python
pymocd.set_thread_count(n)
```

This function should be used before starting the algorithm! (obviously!)

{{< alert context="warning" text="Using set_thread_count again in the samr code has no effect, due to static ThreadPoolBuilder initialization" />}}

{{< alert context="info" text="You need to 'free' the library from memory, to setup a new number of logical threads again." />}}

----

| Parameter     | Type               | Description                                                                    |
|---------------|--------------------|--------------------------------------------------------------------------------|
| **n**         | `u16`              | How many logical threads you want to use.                                      |