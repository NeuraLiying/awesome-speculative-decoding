# Awesome Speculative Decoding

> A curated list of speculative decoding papers, code, and resources for efficient large language model inference.

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

---

## Contents

- [Overview](#overview)
- [Training-based Methods](#training-based-methods)
- [Training-free Methods](#training-free-methods)
- [Implementations](#implementations)
- [Tutorials & Blogs](#tutorials--blogs)
- [Benchmarks & Datasets](#benchmarks--datasets)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Speculative decoding refers to a family of methods that accelerate large language model inference by generating multiple tokens (or hidden states) in advance and verifying or correcting them using the original model, thereby reducing latency and computational cost.

---

## Training-based Methods

<!-- Example entry -->
### [Paper Title](arxiv-link)

- **Authors**: Name1, Name2, Name3, ...
- **arXiv**: [arxiv.org/abs/xxxx.xxxxx](arxiv-link)
- **GitHub**: [github.com/xxx/yyy](github-link) <!-- (optional, if available) -->
- **Webpage/Project**: [project page](url) <!-- (optional, if available) -->
- **Notes**: _Optional: short summary, highlight of key idea, or application._

<!-- Repeat for each training-based method -->
#### Examples:

### [Multi-Token Prediction: Accelerating LLM Decoding with Multi-token Parallelism](https://arxiv.org/abs/2401.10117)
- **Authors**: Fan Zhou, Li Dong, Furu Wei, et al.
- **arXiv**: [2401.10117](https://arxiv.org/abs/2401.10117)
- **GitHub**: [InternLM/MTP](https://github.com/InternLM/MTP)
- **Webpage/Project**: [InternLM-MTP Home](https://github.com/InternLM/MTP)
- **Notes**: Proposes a draft-verifier framework using multi-token prediction for parallel decoding.

### [EAGLE: Early Accept/Reject Generation for Language Model Inference](https://arxiv.org/abs/2310.03713)
- **Authors**: Yongchao Jin, Xiaoyu Chen, et al.
- **arXiv**: [2310.03713](https://arxiv.org/abs/2310.03713)
- **GitHub**: [OpenLMLab/EAGLE](https://github.com/OpenLMLab/EAGLE)
- **Webpage/Project**: [OpenLMLab/EAGLE](https://github.com/OpenLMLab/EAGLE)
- **Notes**: Utilizes token-level accept/reject mechanisms for faster inference.

---

## Training-free Methods

#### Example Entry:

### [Speculative Decoding: Drafting and Verifying Language Model Generations](https://arxiv.org/abs/2302.01318)
- **Authors**: Jacob Austin, Augustus Odena, Maximilian Nickel, et al.
- **arXiv**: [2302.01318](https://arxiv.org/abs/2302.01318)
- **GitHub**: [HuggingFace Transformers PR#21937](https://github.com/huggingface/transformers/pull/21937)
- **Webpage/Project**: [HuggingFace Blog](https://huggingface.co/blog/speculative-decoding)
- **Notes**: The original speculative decoding proposal, using a draft model and verifier to accelerate LLM inference.

<!-- Repeat for each paper -->

---

## Implementations

- [HuggingFace Transformers (Speculative Decoding PR)](https://github.com/huggingface/transformers/pull/21937)
- [InternLM MTP](https://github.com/InternLM/MTP)
- [OpenLMLab/EAGLE](https://github.com/OpenLMLab/EAGLE)
- [More...](#)

---

## Tutorials & Blogs

- [Speculative Decoding in Transformers (HuggingFace blog)](https://huggingface.co/blog/speculative-decoding)
- [EAGLE: A New Path for Fast Language Model Inference](https://zhuanlan.zhihu.com/p/666888691)
- [More...](#)

---

## Benchmarks & Datasets

- [BIG-Bench Hard](https://github.com/google/BIG-bench)
- [MMLU-Pro](https://github.com/haonan-li/mmlu-pro)

---

## Contributing

Contributions are welcome!  
Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on adding new papers, codebases, or tutorials.

---

## License

[CC0-1.0](LICENSE)
