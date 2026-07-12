# Awesome Speculative Decoding [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated, link-verified survey of **speculative decoding** research — draft, verify, accelerate. Continuously updated.

**Total: ~115 papers · last updated 2026-07-13**

Speculative decoding accelerates autoregressive LLM inference by having a cheap **drafter** propose several tokens that the **target** model verifies in a single forward pass. Done with the right (modified-rejection) sampling, the output distribution is **provably identical** to the target's — speedup for free, with no quality loss. This list organizes the field by *where the draft comes from* and *how it is verified*, from the foundational theory through training-based drafters (Medusa, EAGLE), training-free and self-speculative methods, retrieval/n-gram drafters, tree and multi-draft verification, and specialized targets (MoE, multimodal/VLM, speech, diffusion, reasoning/long-context).

## Table of Contents

- [1. Foundations & Theory](#1-foundations--theory)
- [2. Surveys & Benchmarks](#2-surveys--benchmarks)
- [3. Independent Draft Models](#3-independent-draft-models)
- [4. Multi-Head Parallel Decoding (Medusa family & MTP)](#4-multi-head-parallel-decoding-medusa-family--mtp)
- [5. Feature-Level Autoregressive Drafters (EAGLE family)](#5-feature-level-autoregressive-drafters-eagle-family)
- [6. Self-Speculative & Early-Exit Decoding](#6-self-speculative--early-exit-decoding)
- [7. Lookahead & Jacobi Decoding](#7-lookahead--jacobi-decoding)
- [8. Retrieval / Suffix / N-gram Drafting](#8-retrieval--suffix--n-gram-drafting)
- [9. Tree Construction & Multi-Draft Verification](#9-tree-construction--multi-draft-verification)
- [10. Speculation-Length & Budget Control](#10-speculation-length--budget-control)
- [11. Speculative Decoding for MoE](#11-speculative-decoding-for-moe)
- [12. Quantization-Aware & Memory-Efficient SD](#12-quantization-aware--memory-efficient-sd)
- [13. Multimodal, VLM & Speech](#13-multimodal-vlm--speech)
- [14. Speculative Decoding for Diffusion](#14-speculative-decoding-for-diffusion)
- [15. Systems, Serving & Parallelism](#15-systems-serving--parallelism)
- [16. Reasoning & Long-Context SD](#16-reasoning--long-context-sd)
- [17. Speculative Decoding + Other Techniques](#17-speculative-decoding--other-techniques)
- [Production & Open-Source Implementations](#production--open-source-implementations)
- [Related Awesome Lists](#related-awesome-lists)
- [Verification Methodology](#verification-methodology)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## 1. Foundations & Theory

*The origin papers and the theory of lossless speculative sampling — start here.* Ordered chronologically.

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| Blockwise Parallel Decoding for Deep Autoregressive Models | [1811.03115](https://arxiv.org/abs/1811.03115) | / | 2018 | NeurIPS 18 | Earliest predecessor: predict multiple steps in parallel, back off to the longest prefix a scorer accepts. |
| Fast Inference from Transformers via Speculative Decoding | [2211.17192](https://arxiv.org/abs/2211.17192) | / | 2022 | ICML 23 | **The original Google paper.** A draft model proposes tokens the target verifies in parallel; 2–3× lossless. |
| Accelerating Large Language Model Decoding with Speculative Sampling | [2302.01318](https://arxiv.org/abs/2302.01318) | / | 2023 | DeepMind tech report | **Co-foundational (DeepMind).** Modified rejection sampling preserves the target distribution; 2–2.5× on Chinchilla 70B. |
| Decoding Speculative Decoding | [2402.01528](https://arxiv.org/abs/2402.01528) | / | 2024 | NAACL 25 | Formal analysis/clarification of the speculative decoding framework. |
| A Theoretical Perspective for Speculative Decoding Algorithm | [2411.00841](https://arxiv.org/abs/2411.00841) | / | 2024 | NeurIPS 24 | Markov-chain abstraction deriving output-quality and speedup limits via total-variation distance. |
| Inevitable Trade-off between Watermark Strength and Speculative Sampling Efficiency | [2410.20418](https://arxiv.org/abs/2410.20418) | / | 2024 | NeurIPS 24 | Proves a fundamental tension between watermarking and speculative-sampling efficiency. |
| polybasic Speculative Decoding Through a Theoretical Perspective | [2510.26527](https://arxiv.org/abs/2510.26527) | / | 2025 | ICML 25 | Generalized theoretical treatment of multi-form speculative sampling. |

## 2. Surveys & Benchmarks

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling | [2509.04474](https://arxiv.org/abs/2509.04474) | / | 2025 | ICLR 26 | Systematic benchmark of SD methods for test-time-scaling workloads. |
| Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding | [2401.07851](https://arxiv.org/abs/2401.07851) | / | 2024 | ACL 24 Findings | Broad survey taxonomizing SD approaches. |

## 3. Independent Draft Models

*A separate, smaller drafter model proposes tokens; the target verifies. Includes drafter training, distillation, and cascades.*

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| Mamba Drafters for Speculative Decoding | [2506.01206](https://arxiv.org/abs/2506.01206) | / | 2025 | EMNLP 25 Findings | Uses efficient Mamba/SSM state-space models as drafters. |
| Training Domain Draft Models for Speculative Decoding | [2503.07807](https://arxiv.org/abs/2503.07807) | / | 2025 | arXiv | Best practices for domain-specific drafters (function-call, biology, Chinese). |
| Cascade Speculative Drafting for Even Faster LLM Inference | [2312.11462](https://arxiv.org/abs/2312.11462) | / | 2024 | NeurIPS 24 | Vertical & horizontal cascades of drafters of increasing size. |
| Online Speculative Decoding | [2310.07177](https://arxiv.org/abs/2310.07177) | [repo](https://github.com/LiuXiaoxuanPKU/OSD) | 2024 | ICML 24 | Continuously retrains the drafter on live queries via online KD. |
| DistillSpec: Improving Speculative Decoding via Knowledge Distillation | [2310.08461](https://arxiv.org/abs/2310.08461) | / | 2024 | ICLR 24 | On-policy KD aligning drafter to target; 10–45% over vanilla SD. |
| Speculative Decoding with Big Little Decoder (BiLD) | [2302.07863](https://arxiv.org/abs/2302.07863) | [repo](https://github.com/kssteven418/biglittledecoder) | 2023 | NeurIPS 23 | Small model decodes, big model refines via fallback/rollback; 2.12×, plug-and-play. |
| Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation (SpecDec) | [2203.16487](https://arxiv.org/abs/2203.16487) | [repo](https://github.com/hemingkx/SpecDec) | 2023 | EMNLP 23 Findings | First to formalize speculative execution for AR decoding; independent Spec-Drafter + Spec-Verification. |

## 4. Multi-Head Parallel Decoding (Medusa family & MTP)

*Add extra prediction heads to the target so it drafts multiple tokens itself in one pass.*

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| Jakiro: Boosting Speculative Decoding with Decoupled Multi-Head via MoE | [2502.06282](https://arxiv.org/abs/2502.06282) | [repo](https://github.com/haiduo/Jakiro) | 2025 | ACL 26 | MoE produces diverse decoupled multi-head candidates + hybrid AR-then-parallel decoding. |
| Clover: Regressive Lightweight Speculative Decoding with Sequential Knowledge | [2405.00263](https://arxiv.org/abs/2405.00263) | / | 2024 | arXiv | Regressive connection + attention decoder injects sequential knowledge; beats Medusa. |
| Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads | [2401.10774](https://arxiv.org/abs/2401.10774) | [repo](https://github.com/FasterDecoding/Medusa) | 2024 | ICML 24 | Extra heads + tree attention; Medusa-1 2.2×, Medusa-2 up to 3.6×. |
| Better & Faster Large Language Models via Multi-token Prediction | [2404.19737](https://arxiv.org/abs/2404.19737) | / | 2024 | ICML 24 | Trains n output heads to predict n future tokens; up to 3× faster (basis of DeepSeek-V3 MTP). |
| Amphista: Bi-directional Multi-head Decoding for Accelerating LLM Inference | [2406.13170](https://arxiv.org/abs/2406.13170) | / | 2024 | NAACL 25 | Auto-embedding block with bi-directional attention so drafting heads interact. |
| Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding | [2402.05109](https://arxiv.org/abs/2402.05109) | [repo](https://github.com/zankner/Hydra) | 2024 | arXiv | Adds autoregressive coupling across Medusa heads (Hydra++). |
| PaSS: Parallel Speculative Sampling | [2311.13581](https://arxiv.org/abs/2311.13581) | / | 2023 | ENLSP-III @ NeurIPS 23 | Single-model parallel drafting via embedding-sized marker tokens; no second model. |

## 5. Feature-Level Autoregressive Drafters (EAGLE family)

*Draft at the hidden-state/feature level — the dominant high-performance family.*

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test | [2503.01840](https://arxiv.org/abs/2503.01840) | [repo](https://github.com/SafeAILab/EAGLE) | 2025 | NeurIPS 25 | Direct token prediction + multi-layer feature fusion; up to 6.5×, ~1.4× over EAGLE-2. |
| Gumiho: A Hybrid Architecture to Prioritize Early Tokens in Speculative Decoding | [2503.10135](https://arxiv.org/abs/2503.10135) | [repo](https://github.com/AMD-AIG-AIMA/Gumiho) | 2025 | ICML 25 | EAGLE for the first two draft tokens, Medusa for the next five. |
| Learning Harmonized Representations for Speculative Sampling (HASS) | [2408.15766](https://arxiv.org/abs/2408.15766) | [repo](https://github.com/HArmonizedSS/HASS) | 2024 | ICLR 25 | Harmonized distillation fixes train/decode mismatch; +8–20% over EAGLE-2. |
| Turning Trash into Treasure: Accelerating Inference of LLMs with Token Recycling | [2408.08696](https://arxiv.org/abs/2408.08696) | [repo](https://github.com/Luowaterbi/TokenRecycling) | 2024 | ACL 25 | Training-free BFS draft tree from recycled rejected candidates; ~2×. |
| BiTA: Bi-Directional Tuning for Lossless Acceleration in LLMs | [2401.12522](https://arxiv.org/abs/2401.12522) | [repo](https://github.com/linfeng93/BiTA) | 2024 | ESWA 25 | PEFT bi-directional tuning; semi-AR drafting + tree decoding; 2.7× lossless. |
| EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees | [2406.16858](https://arxiv.org/abs/2406.16858) | [repo](https://github.com/SafeAILab/EAGLE) | 2024 | EMNLP 24 | Context-aware dynamic draft tree pruned by calibrated confidence; 3.05–4.26×. |
| GliDe with a CaPE: A Low-Hassle Method to Accelerate Speculative Decoding | [2402.02082](https://arxiv.org/abs/2402.02082) | / | 2024 | ICML 24 | Drafter reuses the target's KV cache; CaPE expands proposals by confidence. |
| EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty | [2401.15077](https://arxiv.org/abs/2401.15077) | [repo](https://github.com/SafeAILab/EAGLE) | 2024 | ICML 24 | Feature-level autoregressive drafter; 2.7–3.5× lossless. |

## 6. Self-Speculative & Early-Exit Decoding

*No separate drafter — the target model drafts itself by exiting early / skipping layers.*

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration | [2410.06916](https://arxiv.org/abs/2410.06916) | [repo](https://github.com/hemingkx/SWIFT) | 2024 | ICLR 25 | Adaptive on-the-fly self-speculative decoding. |
| SPACE (SMART): Generation Meets Verification via Smart Parallel Auto-Correct Decoding | [2402.11809](https://arxiv.org/abs/2402.11809) | [repo](https://github.com/cteant/SPACE) | 2024 | ACL 24 Findings | One model does multi-token prediction + interleaved auto-correct verify. |
| Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting | [2404.18911](https://arxiv.org/abs/2404.18911) | [repo](https://github.com/Equationliu/Kangaroo) | 2024 | NeurIPS 24 | Fixed shallow sub-network + adapter drafter, with a confidence-based second early exit. |
| Speculative Decoding via Early-exiting with Thompson Sampling Control | [2406.03853](https://arxiv.org/abs/2406.03853) | / | 2024 | ACL 24 Findings | Trainable early-exit layer on top of the N-th hidden state; Thompson-sampling control. |
| LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding | [2404.16710](https://arxiv.org/abs/2404.16710) | [repo](https://github.com/facebookresearch/LayerSkip) | 2024 | ACL 24 | Trains with layer dropout + early-exit loss; early layers draft, rest verify. |
| Draft & Verify: Lossless LLM Acceleration via Self-Speculative Decoding | [2309.08168](https://arxiv.org/abs/2309.08168) | / | 2023 | ACL 24 | Drafts by skipping intermediate layers, verifies with the full LLM. |

## 7. Lookahead & Jacobi Decoding

*Training-free parallel decoding from Jacobi iteration / n-gram pools.*

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| Lossless Acceleration of LLM via Adaptive N-gram Parallel Decoding (ANPD) | [2404.08698](https://arxiv.org/abs/2404.08698) | / | 2024 | NAACL Industry 24 | Two-stage adaptive context-aware n-gram drafting; up to 3.67× lossless. |
| Break the Sequential Dependency of LLM Inference Using Lookahead Decoding | [2402.02057](https://arxiv.org/abs/2402.02057) | [repo](https://github.com/hao-ai-lab/LookaheadDecoding) | 2024 | ICML 24 | Training-free Jacobi n-gram pool mined from iteration trajectories; single-pass verify. |

## 8. Retrieval / Suffix / N-gram Drafting

*Draft tokens by looking them up in a datastore, suffix tree, or in-context reference — fully training-free.*

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| LogitSpec: Accelerating Retrieval-based SD via Next-Next-Token Speculation | [2507.01449](https://arxiv.org/abs/2507.01449) | [repo](https://github.com/smart-lty/LogitSpec) | 2025 | arXiv | Uses the last-token logit to speculate next-next token, broadening retrieval queries; 2.61×. |
| RASD: Retrieval-Augmented Speculative Decoding | [2503.03434](https://arxiv.org/abs/2503.03434) | / | 2025 | ACL 25 Findings | Fuses a draft-model tree with a retrieval tree via probability pruning + longest-prefix match. |
| DReSD: Dense Retrieval for Speculative Decoding | [2502.15572](https://arxiv.org/abs/2502.15572) | / | 2025 | ACL 25 Findings | Dense ANN retrieval over contextualised token embeddings; +87% acceptance vs REST. |
| SAM Decoding: Speculative Decoding via Suffix Automaton | [2411.10666](https://arxiv.org/abs/2411.10666) | [repo](https://github.com/hyx1999/SAM-Decoding) | 2024 | ACL 25 | On-the-fly suffix-automaton n-gram drafter. |
| SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications | [2411.04975](https://arxiv.org/abs/2411.04975) | [repo](https://github.com/snowflakedb/ArcticInference) | 2024 | NeurIPS 25 | Suffix-tree cache of long sequences from prompts/outputs; 5.3× on agentic workloads. |
| Nearest Neighbor Speculative Decoding for LLM Generation and Attribution (NEST) | [2405.19325](https://arxiv.org/abs/2405.19325) | [repo](https://github.com/facebookresearch/NEST) | 2024 | NeurIPS 24 | Token-level kNN retrieval proposes spans; semi-parametric, training-free. |
| TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical SD | [2404.11912](https://arxiv.org/abs/2404.11912) | [repo](https://github.com/Infini-AI-Lab/TriForce) | 2024 | COLM 24 | Hierarchical tree SD using a retrieval-based sparse-KV drafter; scales to long context. |
| Ouroboros: Generating Longer Drafts Phrase by Phrase for Faster SD | [2402.13720](https://arxiv.org/abs/2402.13720) | [repo](https://github.com/thunlp/Ouroboros) | 2024 | EMNLP 24 | Extends the last draft token to phrases; training-free. |
| REST: Retrieval-Based Speculative Decoding | [2311.08252](https://arxiv.org/abs/2311.08252) | [repo](https://github.com/FasterDecoding/REST) | 2024 | NAACL 24 | Drafts from a retrieval datastore over existing text; plug-and-play. |
| Inference with Reference: Lossless Acceleration of LLMs (LLMA) | [2304.04487](https://arxiv.org/abs/2304.04487) | [repo](https://github.com/microsoft/LMOps) | 2023 | arXiv | Copies & verifies spans from in-context references; >2× for RAG/search/multi-turn. |

## 9. Tree Construction & Multi-Draft Verification

*How to shape the draft tree and how to verify many drafts at once.*

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| Yggdrasil: Bridging Dynamic Speculation and Static Runtime for Latency-Optimal Tree-Based Decoding | [2512.23858](https://arxiv.org/abs/2512.23858) | / | 2025 | NeurIPS 25 | Unifies dynamic speculation with a static runtime for latency-optimal tree decoding. |
| HeteroSpec: Leveraging Contextual Heterogeneity for Efficient Speculative Decoding | [2505.13254](https://arxiv.org/abs/2505.13254) | / | 2025 | arXiv | Dynamically allocates tree depth/branches by linguistic-context complexity. |
| TETRIS: Optimal Draft Token Selection for Batch Speculative Decoding | [2502.15197](https://arxiv.org/abs/2502.15197) | / | 2025 | ACL 25 | Selects draft tokens for verification by drafter probability, for batched SD. |
| Judge Decoding: Faster Speculative Sampling Requires Going Beyond Model Alignment | [2501.19309](https://arxiv.org/abs/2501.19309) | / | 2025 | ICLR 25 | A separate judge verifier goes beyond simple draft-target alignment. |
| Towards Optimal Multi-draft Speculative Decoding | [2502.18779](https://arxiv.org/abs/2502.18779) | / | 2025 | ICLR 25 | Characterises the optimum over multiple drafts. |
| Multi-Draft Speculative Sampling: Canonical Decomposition and Theoretical Limits | [2410.18234](https://arxiv.org/abs/2410.18234) | / | 2024 | ICLR 25 | Canonical decomposition + theoretical limits of multi-draft sampling. |
| Block Verification Accelerates Speculative Decoding | [2403.10444](https://arxiv.org/abs/2403.10444) | / | 2024 | ICLR 25 | Verifies draft tokens in blocks for higher throughput. |
| Graph-Structured Speculative Decoding | [2407.16207](https://arxiv.org/abs/2407.16207) | / | 2024 | ACL 24 Findings | Generalises the draft tree to a directed acyclic graph. |
| OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure | [2406.17276](https://arxiv.org/abs/2406.17276) | [repo](https://github.com/Jikai0Wang/OPT-Tree) | 2024 | TACL | Searches the tree maximising expected acceptance length each step. |
| SpecHub: Provable Acceleration to Multi-Draft Speculative Decoding | [2411.05289](https://arxiv.org/abs/2411.05289) | [repo](https://github.com/MasterGodzilla/Speculative_decoding_OT) | 2024 | EMNLP 24 | Provable speed-up for multi-draft verification. |
| Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding | [2402.12374](https://arxiv.org/abs/2402.12374) | [repo](https://github.com/infini-ai-lab/sequoia) | 2024 | NeurIPS 24 | DP-optimal token tree + hardware-aware size/depth optimizer; 4.04×. |
| SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference | [2305.09781](https://arxiv.org/abs/2305.09781) | [repo](https://github.com/flexflow/FlexFlow) | 2024 | ASPLOS 24 | Seminal tree-based SD system; drafters predict into a token tree verified in parallel; 1.5–3.5×. |
| SpecTr: Fast Speculative Decoding via Optimal Transport | [2310.15141](https://arxiv.org/abs/2310.15141) | / | 2023 | NeurIPS 23 | Multi-draft selection formulated as optimal transport, solved near-linearly. |

## 10. Speculation-Length & Budget Control

*Adaptively decide how many tokens to draft and when to stop.*

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms | [2505.15141](https://arxiv.org/abs/2505.15141) | / | 2025 | ICML 25 | Treats draft-length selection as a multi-armed bandit. |
| AdaSpec: Adaptive Speculative Decoding for Fast, SLO-Aware LLM Serving | [2503.05096](https://arxiv.org/abs/2503.05096) | [repo](https://github.com/cerebellumking/AdaSpec) | 2025 | arXiv | SLO-aware adaptive speculation length for serving. |
| AdaEAGLE: Optimizing SD via Explicit Modeling of Adaptive Draft Structures | [2412.18910](https://arxiv.org/abs/2412.18910) | / | 2024 | arXiv | Trains an MLP to predict next-round draft length within EAGLE. |
| Draft Model Knows When to Stop: Self-Verification SD for Long-Form Generation | [2411.18462](https://arxiv.org/abs/2411.18462) | / | 2024 | EMNLP 25 | Entropy-based self-verification length policy; plug-and-play. |
| Optimizing Speculative Decoding for Serving LLMs Using Goodput (SmartSpec/TurboSpec) | [2406.14066](https://arxiv.org/abs/2406.14066) | / | 2024 | arXiv | Per-request speculation length via a goodput metric; up to 3.2×. |
| Dynamic Depth Decoding: Faster Speculative Decoding for LLMs | [2409.00142](https://arxiv.org/abs/2409.00142) | / | 2024 | arXiv | Beam-probability heuristic for when to stop drafting (EAGLE framework). |
| Dynamic Speculation Lookahead Accelerates Speculative Decoding | [2405.04304](https://arxiv.org/abs/2405.04304) | / | 2024 | arXiv | An FFN decides whether to keep drafting from confidence/entropy/position. |

## 11. Speculative Decoding for MoE

*MoE breaks SD's "verification ≈ 1 token" assumption — these methods adapt SD to mixture-of-experts.*

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| SpecMoE: A Fast and Efficient MoE Inference via Self-Assisted Speculative Decoding | [2604.10152](https://arxiv.org/abs/2604.10152) | / | 2026 | DAC 26 | Self-assisted SD for CPU-offloaded MoE; up to 4.30× throughput. |
| Speculating Experts Accelerates Inference for Mixture-of-Experts | [2603.19289](https://arxiv.org/abs/2603.19289) | [repo](https://github.com/axonn-ai/yalis/tree/offload_prefetch) | 2026 | arXiv | Predicts & prefetches future experts to overlap CPU-GPU transfer. |
| MoE-Spec: Expert Budgeting for Efficient Speculative Decoding | [2602.16052](https://arxiv.org/abs/2602.16052) | / | 2026 | arXiv | Training-free per-layer expert-capacity budgeting; +10–30% over EAGLE-3. |
| SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE Inference | [2510.10302](https://arxiv.org/abs/2510.10302) | / | 2025 | arXiv | SD-aware expert offloading + prefetch + compute-communication pipelining; 1.07–3.5×. |
| Utility-Driven Speculative Decoding for Mixture-of-Experts (Cascade) | [2506.20675](https://arxiv.org/abs/2506.20675) | / | 2025 | arXiv | Speculation-utility draft-length tuning; implemented in vLLM. |
| MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE | [2505.19645](https://arxiv.org/abs/2505.19645) | / | 2025 | NeurIPS 25 | MoE benefits more from SD than dense models; target-efficiency metric; 2.29×. |

## 12. Quantization-Aware & Memory-Efficient SD

*Combine SD with low-precision weights/activations and KV-cache compression.*

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| Quasar: Quantized Self-Speculative Acceleration via Memory-Efficient Verification | [2603.01399](https://arxiv.org/abs/2603.01399) | [repo](https://github.com/Tom-HG/Quasar) | 2026 | arXiv | Low-bit quantization on the verification stage of self-spec SD; 1.28×. |
| SPEQ (From Quarter to All): FP Exponent Remapping & Parameter Sharing | [2510.18525](https://arxiv.org/abs/2510.18525) | / | 2025 | arXiv | Algorithm-hardware co-design forms the drafter from part of the weights; 2.07×. |
| Speculative Decoding Meets Quantization (SpecMQuant) | [2505.22179](https://arxiv.org/abs/2505.22179) | [repo](https://github.com/AI9Stars/SpecMQuant) | 2025 | arXiv | Hierarchical tree→sequence drafts on 4-bit Llama-3-70B; 2.78×. |
| ML-SpecQD: Multi-Level Speculative Decoding with Quantized Drafts | [2503.13565](https://arxiv.org/abs/2503.13565) | / | 2025 | arXiv | MXFP4 direct-cast quantized drafts + multi-level speculation; 2.72×. |
| QSpec: Speculative Decoding with Complementary Quantization Schemes | [2410.11305](https://arxiv.org/abs/2410.11305) | [repo](https://github.com/hku-netexplo-lab/QSpec) | 2024 | EMNLP 25 | Low-precision draft, high-precision verify; up to 1.64× plug-and-play. |

## 13. Multimodal, VLM & Speech

*Speculative decoding for vision-language models, video LLMs, VLAs, ASR, and TTS.*

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| Speech Speculative Decoding (SSD): Accelerating Autoregressive Speech Synthesis | [2505.15380](https://arxiv.org/abs/2505.15380) | / | 2025 | INTERSPEECH 25 | Lightweight TTS drafter verified in parallel; 1.4× with high fidelity. |
| SpecASR: Accelerating LLM-based ASR via Speculative Decoding | [2507.18181](https://arxiv.org/abs/2507.18181) | / | 2025 | DAC 25 | Audio-conditioned SD with adaptive length + sparse token trees; 3.04–3.79×. |
| Spec-VLA: Speculative Decoding for Vision-Language-Action Models with Relaxed Acceptance | [2507.22424](https://arxiv.org/abs/2507.22424) | / | 2025 | EMNLP 25 | First SD for VLA; relative-distance relaxed acceptance; 1.42× over OpenVLA. |
| ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding | [2509.15235](https://arxiv.org/abs/2509.15235) | [repo](https://github.com/KangJialiang/ViSpec) | 2025 | NeurIPS 25 | Lightweight vision adapter handles redundant image tokens; 3.2× on VLMs. |
| SpecVLM: Fast Speculative Decoding in Vision-Language Models (EagleVLM) | [2509.11815](https://arxiv.org/abs/2509.11815) | [repo](https://github.com/haiduo/SpecVLM) | 2025 | arXiv | EAGLE-2-style VLM drafter + elastic visual compressor; 1.5–2.3×. |
| SpecVLM: Enhancing SD of Video LLMs via Verifier-Guided Token Pruning | [2508.16201](https://arxiv.org/abs/2508.16201) | [repo](https://github.com/zju-jiyicheng/SpecVLM) | 2025 | EMNLP 25 | Training-free; prunes up to 90% video tokens for the draft; 2.68×. |
| DREAM: Drafting with Refined Target Features for Multimodal Speculative Decoding | [2505.19201](https://arxiv.org/abs/2505.19201) | [repo](https://github.com/SAI-Lab-NYU/DREAM) | 2025 | NeurIPS 25 | Entropy-adaptive cross-attention + visual compression; 3.6× across LLaVA/Pixtral/Gemma3. |
| Speculative Decoding Reimagined for Multimodal LLMs (MSD) | [2505.14260](https://arxiv.org/abs/2505.14260) | [repo](https://github.com/Lyn-Lucy/MSD) | 2025 | arXiv | Decouples text vs visual tokens in the drafter; 2.46× on LLaVA-1.5. |
| On Speculative Decoding for Multimodal Large Language Models (CMSD) | [2404.08856](https://arxiv.org/abs/2404.08856) | / | 2024 | CVPR 24 Workshop | Seminal; up to 2.37× memory-bound speedup for MLLMs with a tiny text drafter. |
| Whisper in Medusa's Ear: Multi-head Efficient Decoding for Transformer-based ASR | [2409.15869](https://arxiv.org/abs/2409.15869) | / | 2024 | arXiv | Multi-token Whisper; ~50% ASR/translation latency cut. |

## 14. Speculative Decoding for Diffusion

*Apply speculative/parallel decoding to autoregressive visual generation and diffusion models.*

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| Diffusion Models are Secretly Exchangeable: Parallelizing DDPMs via Autospeculation | [2505.03983](https://arxiv.org/abs/2505.03983) | / | 2025 | ICML 25 | Exploits exchangeability to parallelise DDPM sampling via auto-speculation. |
| Accelerated Diffusion Models via Speculative Sampling | [2501.05370](https://arxiv.org/abs/2501.05370) | / | 2025 | ICML 25 | Speculative sampling for diffusion-model acceleration. |
| Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion (SpecDiff) | [2408.05636](https://arxiv.org/abs/2408.05636) | / | 2024 | NAACL 25 | Uses a discrete-diffusion drafter so drafting & verification both parallelise; 7.2×. |
| LANTERN: Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding | [2410.03355](https://arxiv.org/abs/2410.03355) | [repo](https://github.com/jadohu/LANTERN) | 2024 | ICLR 25 | Relaxed SD for visual AR generation. |
| Accelerating Auto-regressive T2I with Training-free Speculative Jacobi Decoding (SJD) | [2410.01699](https://arxiv.org/abs/2410.01699) | [repo](https://github.com/tyshiwo1/Accelerating-T2I-AR-with-SJD) | 2024 | ICLR 25 | Training-free probabilistic parallel decoding for AR text-to-image; preserves diversity. |

## 15. Systems, Serving & Parallelism

*Making SD work in real serving stacks — batching, multi-GPU, heterogeneous hardware, consumer devices.*

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| SpecBranch: Speculative Decoding via Hybrid Drafting and Branch Parallelism | [2506.01979](https://arxiv.org/abs/2506.01979) | / | 2025 | ICLR 26 | Hybrid drafting + rollback-aware branch parallelism. |
| OmniDraft: A Cross-vocabulary, Online Adaptive Drafter for On-device SD | [2507.02659](https://arxiv.org/abs/2507.02659) | / | 2025 | NeurIPS 25 | Cross-vocabulary online-adaptive drafter for on-device inference. |
| Dovetail: A CPU/GPU Heterogeneous Speculative Decoding for LLM Inference | [2412.18934](https://arxiv.org/abs/2412.18934) | / | 2024 | EMNLP 25 | Heterogeneous CPU/GPU SD. |
| EasySpec: Layer-Parallel Speculative Decoding for Efficient Multi-GPU Utilization | [2502.02493](https://arxiv.org/abs/2502.02493) | [repo](https://github.com/Yize-Wu/EasySpec) | 2025 | NeurIPS 25 | Layer-parallel SD for multi-GPU. |
| Speculative Streaming: Fast LLM Inference without Auxiliary Models | [2402.11131](https://arxiv.org/abs/2402.11131) | / | 2024 | NeurIPS 24 | Fuses drafting into the target via n-gram prediction; ~10000× fewer params than Medusa. |
| PEARL: Parallel Speculative Decoding with Adaptive Draft Length | [2408.11850](https://arxiv.org/abs/2408.11850) | [repo](https://github.com/smart-lty/ParallelSpeculativeDecoding) | 2024 | ICLR 25 | Pre/post-verify overlaps drafting and verification. |
| MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation | [2408.11049](https://arxiv.org/abs/2408.11049) | [repo](https://github.com/Infini-AI-Lab/MagicDec) | 2024 | ICLR 25 | SD helps even at high batch for long context via a sparse-KV drafter. |
| Distributed Speculative Inference (DSI): Speculation Parallelism | [2405.14105](https://arxiv.org/abs/2405.14105) | / | 2024 | ICLR 25 | Speculation parallelism; provably faster lossless inference. |
| SpecExec: Massively Parallel Speculative Decoding for Consumer Devices | [2406.02532](https://arxiv.org/abs/2406.02532) | [repo](https://github.com/yandex-research/specexec) | 2024 | NeurIPS 24 | Large cache tree validates up to 20 tokens/pass; 50B+ on consumer GPUs. |
| BASS: Batched Attention-optimized Speculative Sampling | [2404.15778](https://arxiv.org/abs/2404.15778) | / | 2024 | ACL 24 Findings | Batched, attention-optimized spec sampling for serving. |

## 16. Reasoning & Long-Context SD

*Speculative decoding for reasoning models, CoT, and test-time scaling.*

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| FastGRPO: Accelerating Policy Optimization via Concurrency-aware SD & Online Draft Learning | [2509.21792](https://arxiv.org/abs/2509.21792) | [repo](https://github.com/yedaotian9/GRPO_speculative) | 2025 | ICLR 26 | SD-accelerated policy optimisation with online draft learning. |
| Scaling Speculative Decoding with Lookahead Reasoning | [2506.19830](https://arxiv.org/abs/2506.19830) | [repo](https://github.com/hao-ai-lab/LookaheadReasoning) | 2025 | NeurIPS 25 | Scales SD by speculatively drafting reasoning steps ahead. |
| Accelerating LLM Reasoning via Speculative Search | [2505.02865](https://arxiv.org/abs/2505.02865) | / | 2025 | ICML 25 | Combines speculative decoding with search for reasoning. |
| Speculative Prefill: Turbocharging TTFT with Token-Importance Estimation | [2502.02789](https://arxiv.org/abs/2502.02789) | / | 2025 | ICML 25 | Training-free acceleration of time-to-first-token. |
| Reward-Guided Speculative Decoding for Efficient LLM Reasoning | [2501.19324](https://arxiv.org/abs/2501.19324) | [repo](https://github.com/BaohaoLiao/RSD) | 2025 | ICML 25 | Uses a reward model to guide speculative drafts for reasoning. |

## 17. Speculative Decoding + Other Techniques

*SD combined with RAG, contrastive decoding, attention variants, constrained decoding, and agentic loops.*

| Title | arXiv | GitHub | Year | Venue | One-line |
|-------|-------|--------|------|-------|----------|
| Speculative Actions: A Lossless Framework for Faster Agentic Systems | [2510.04371](https://arxiv.org/abs/2510.04371) | / | 2025 | ICLR 26 | Lossless speculative execution of agentic tool actions. |
| Constrained Decoding with Speculative Lookaheads | [2412.10418](https://arxiv.org/abs/2412.10418) | / | 2024 | NAACL 25 | Uses speculative lookahead to accelerate grammar/vocab-constrained decoding. |
| Mixture of Attentions For Speculative Decoding | [2410.03804](https://arxiv.org/abs/2410.03804) | / | 2024 | ICLR 25 | Mixed attention patterns to improve drafting/verification. |
| Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting | [2407.08223](https://arxiv.org/abs/2407.08223) | / | 2024 | ICLR 25 | Drafts multiple RAG answers in parallel and verifies. |
| Speculative Contrastive Decoding | [2311.08981](https://arxiv.org/abs/2311.08981) | / | 2023 | ACL 24 | Combines SD with contrastive decoding for both speed and quality. |

---

## Production & Open-Source Implementations

Speculative decoding is now built into the major inference engines — these are the shipped artifacts, not papers:

- **vLLM** — `Speculative Decoding` support: Eagle / Medusa / EAGLE-2 / draft-model / prompt-lookup / MTP / layer-Skip. [docs](https://docs.vllm.ai/en/latest/features/spec_decode.html) · [repo](https://github.com/vllm-project/vllm)
- **NVIDIA TensorRT-LLM** — native speculative decoding (EAGLE, Medusa, look-ahead, draft-target). [docs](https://nvidia.github.io/TensorRT-LLM/speculative-decoding/) · [repo](https://github.com/NVIDIA/TensorRT-LLM)
- **HuggingFace Transformers / TGI** — `assistant_model` speculative decoding & Medusa. [docs](https://huggingface.co/docs/transformers/en/llm_tutorial_optimization) · [TGI](https://github.com/huggingface/text-generation-inference)
- **DeepSpeed-FastGen** — Hybrid engine with high-throughput speculative decoding. [repo](https://github.com/microsoft/DeepSpeed)
- **SGLang / LMDeploy / MLC-LLM** — production runtimes with SD support. [SGLang](https://github.com/sgl-project/sglang) · [LMDeploy](https://github.com/InternLM/lmdeploy)

## Related Awesome Lists

- [hemingkx/SpeculativeDecodingPapers](https://github.com/hemingkx/SpeculativeDecodingPapers) — must-read papers & blogs on SD.
- [Geralt-Targaryen/Awesome-Speculative-Decoding](https://github.com/Geralt-Targaryen/Awesome-Speculative-Decoding) — reading notes, organised by venue and method (the bibliography cross-referenced while building this list).
- [xlite-dev/Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference) — broader LLM/VLM inference (attention, quantization, parallelism).
- [wang2226/Awesome-LLM-Decoding](https://github.com/wang2226/Awesome-LLM-Decoding) — decoding methods for foundation models.

---

## Verification Methodology

Every entry was checked for link/title correctness (2026-07-13):

- **arXiv IDs**: ~115 papers were verified by a 24-agent pipeline that opens each `arxiv.org/abs/<id>` page, confirms it resolves, and checks the title matches (modulo casing and method-name prefixes). **Every listed ID resolves.**
- **GitHub links**: the official repo is linked only when it appears on the arXiv abstract page or a verified project site; `/` means no public code was found (never fabricated).
- **Corrections caught**: `2503.05096` is **AdaSpec** (not "SpecServe" — an outdated name); `2506.14066` was retitled to *TurboSpec* in v3 (kept its original v1 title); Amphista/HASS/SWIFT/etc. had minor title normalization. The EAGLE/Medusa/SpecInfer/REST/Lookahead IDs all match the canonical papers.

If a link has rotted since, please open an issue/PR.

## Contributing

PRs welcome. A paper qualifies if **speculative drafting or its verification is a first-class contribution**. Submissions must include a verifiable arXiv ID (must resolve) or a canonical URL — fabricated links will be rejected. Add the row in the right section, sorted by year descending. See [CONTRIBUTING.md](./CONTRIBUTING.md).

## Citation

If you find this list useful, please cite:

```bibtex
@misc{awesome-speculative-decoding-2026,
  author={Ying Li},
  title={Awesome Speculative Decoding: A Curated, Link-Verified Survey},
  year={2026},
  url={https://github.com/NeuraLiying/awesome-speculative-decoding}
}
```

## License

[MIT](./LICENSE).
