# Citations & credits

This framework builds on published benchmarks, datasets, and open-source
libraries. If you use a benchmark through this framework, please cite the
original work as well.

## Benchmarks and datasets

Upstream datasets are **downloaded on first use**; apart from the small test
excerpts listed below, none is part of this repository. Each remains under its
own license, listed below; check it before using a dataset commercially.

| Benchmark | Paper | Dataset | License |
|---|---|---|---|
| [LoCoMo](benchmarks/locomo.md) | Maharana et al., 2024 ([arXiv:2402.17753](https://arxiv.org/abs/2402.17753)) | [`snap-research/locomo`](https://github.com/snap-research/locomo) | CC BY-NC 4.0 (non-commercial) |
| [BEAM](benchmarks/beam.md) | Tavakoli et al., 2025 ([arXiv:2510.27246](https://arxiv.org/abs/2510.27246)) | [`Mohammadta/BEAM`](https://huggingface.co/datasets/Mohammadta/BEAM), [`Mohammadta/BEAM-10M`](https://huggingface.co/datasets/Mohammadta/BEAM-10M) | CC BY-SA 4.0 |
| [MPBench](benchmarks/mpbench.md) | Dash et al., 2026 ([arXiv:2606.04329](https://arxiv.org/abs/2606.04329); AIWILD workshop at ICML 2026) | [`Digital-Trust-Lab/mp-bench`](https://github.com/Digital-Trust-Lab/mp-bench) | Apache-2.0 |
| [LongMemEval](benchmarks/long_mem_eval.md) | Wu et al., 2024 ([arXiv:2410.10813](https://arxiv.org/abs/2410.10813)) | [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | MIT |
| [Evo-Memory (single-turn)](benchmarks/evomemory_single_turn.md) | Wei et al., 2025 ([arXiv:2511.20857](https://arxiv.org/abs/2511.20857)) | Built from the four datasets below | See below |
| [Contextual Integrity](benchmarks/contextual_integrity.md) | Lan et al., 2025 ([arXiv:2506.04245](https://arxiv.org/abs/2506.04245)) | [`huseyinatahaninan/ContextualIntegritySyntheticDataset`](https://huggingface.co/datasets/huseyinatahaninan/ContextualIntegritySyntheticDataset) | Apache-2.0 |
| [STALE](benchmarks/stale.md) | Chao et al., 2026 ([arXiv:2605.06527](https://arxiv.org/abs/2605.06527)) | [`STALEproj/STALE`](https://huggingface.co/datasets/STALEproj/STALE) (code: [`icedreamc/STALE`](https://github.com/icedreamc/STALE)) | CC BY 4.0 (code MIT) |
| [Erasability & Editability](benchmarks/erasability_editability.md) | MEME (paper under review) | [`meme-benchmark/MEME`](https://huggingface.co/datasets/meme-benchmark/MEME) (code: [`SeokwonJung-Jay/MEME-public`](https://github.com/SeokwonJung-Jay/MEME-public)) | CC BY 4.0 (code MIT) |
| [Transparency](benchmarks/transparency.md) | — <!-- Add the project whitepaper once it is published. --> | BEAM chats (above), or our own `datasets/transparency/` | BEAM: CC BY-SA 4.0; ours: CC BY 4.0 |
| [Retention policy](benchmarks/retention_policy.md) | — <!-- Add the project whitepaper once it is published. --> | Our own `datasets/retention_policy/` | CC BY 4.0 |

The Evo-Memory single-turn streams are built from:

| Source | Paper | Dataset | License |
|---|---|---|---|
| AIME 2024 | none | [`HuggingFaceH4/aime_2024`](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) | Not stated on the dataset card |
| GPQA (diamond) | Rein et al., 2023 ([arXiv:2311.12022](https://arxiv.org/abs/2311.12022)) | [`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa) (gated: accept its terms on Hugging Face) | CC BY 4.0 |
| MMLU-Pro | Wang et al., 2024 ([arXiv:2406.01574](https://arxiv.org/abs/2406.01574)) | [`TIGER-Lab/MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | MIT |
| ToolBench stream (BFCL) | Patil et al., 2025 ([ICML 2025](https://proceedings.mlr.press/v267/patil25a.html)); Gorilla, Patil et al., 2023 ([arXiv:2305.15334](https://arxiv.org/abs/2305.15334)) | [`ShishirPatil/gorilla`](https://github.com/ShishirPatil/gorilla) (`berkeley-function-call-leaderboard`) | Apache-2.0 |

### Excerpts in the test fixtures

Three test fixtures under `python_package/tests/fixtures/smoke_e2e/` contain
small excerpts of upstream data, redistributed under the original licenses:

- `stale/smoke.json`: one STALE record (CC BY 4.0), whose distractor sessions
  derive from LongMemEval (MIT)
- `erasability_editability/smoke.json`: a shortened copy of one MEME episode,
  `pl_001` (CC BY 4.0)
- `contextual_integrity/smoke.json`: two samples converted from the first two
  rows of the Contextual Integrity synthetic dataset (Apache-2.0)

The other fixtures are written by hand.

## Libraries and memory systems

| Project | Used for | Paper | License |
|---|---|---|---|
| [tinyagent](https://github.com/mozilla-ai/tinyagent) (Mozilla.ai) | The benchmark agent | none | Apache-2.0 |
| [any-llm](https://github.com/mozilla-ai/any-llm) (Mozilla.ai) | Model access for agents and judges | none | Apache-2.0 |
| [Mem0](https://github.com/mem0ai/mem0) | `mem0` memory plugin | Chhikara et al., 2025 ([arXiv:2504.19413](https://arxiv.org/abs/2504.19413)) | Apache-2.0 |
| [Letta](https://github.com/letta-ai/letta) (MemGPT) | `letta` memory plugin | Packer et al., 2023 ([arXiv:2310.08560](https://arxiv.org/abs/2310.08560)) | Apache-2.0 |
| [Temml](https://github.com/ronkok/Temml) (Ron Kok) | LaTeX rendering in the results browser; bundled, unmodified, under `src/heams/results_browser/static/vendor/` | none | MIT |
| [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) | `hipporag` memory plugin (HippoRAG 2) | Gutiérrez et al., 2024 ([arXiv:2405.14831](https://arxiv.org/abs/2405.14831)); 2025 ([arXiv:2502.14802](https://arxiv.org/abs/2502.14802)) | MIT |

## BibTeX

```bibtex
@article{maharana2024locomo,
  title   = {Evaluating Very Long-Term Conversational Memory of {LLM} Agents},
  author  = {Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal = {arXiv preprint arXiv:2402.17753},
  year    = {2024}
}

@article{tavakoli2025beam,
  title   = {Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in {LLMs}},
  author  = {Tavakoli, Mohammad and Salemi, Alireza and Ye, Carrie and Abdalla, Mohamed and Zamani, Hamed and Mitchell, J Ross},
  journal = {arXiv preprint arXiv:2510.27246},
  year    = {2025}
}

@inproceedings{dash2026mpbench,
  title     = {From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in {LLM} Agents},
  author    = {Dash, Pritam and Ge, Tongyu and Jain, Aditi and Shah, Tanmay and Shang, Zhiwei},
  booktitle = {Second Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD) at ICML 2026},
  year      = {2026},
  eprint    = {2606.04329},
  archivePrefix = {arXiv}
}

@article{wu2024longmemeval,
  title   = {{LongMemEval}: Benchmarking Chat Assistants on Long-Term Interactive Memory},
  author  = {Wu, Di and Wang, Hongwei and Yu, Wenhao and Zhang, Yuwei and Chang, Kai-Wei and Yu, Dong},
  journal = {arXiv preprint arXiv:2410.10813},
  year    = {2024}
}

@article{wei2025evomemory,
  title   = {{Evo-Memory}: Benchmarking {LLM} Agent Test-time Learning with Self-Evolving Memory},
  author  = {Wei, Tianxin and Sachdeva, Noveen and Coleman, Benjamin and He, Zhankui and Bei, Yuanchen and Ning, Xuying and Ai, Mengting and Li, Yunzhe and He, Jingrui and Chi, Ed H. and Wang, Chi and Chen, Shuo and Pereira, Fernando and Kang, Wang-Cheng and Cheng, Derek Zhiyuan},
  journal = {arXiv preprint arXiv:2511.20857},
  year    = {2025}
}

@article{lan2025contextual,
  title   = {Contextual Integrity in {LLMs} via Reasoning and Reinforcement Learning},
  author  = {Lan, Guangchen and Inan, Huseyin A. and Abdelnabi, Sahar and Kulkarni, Janardhan and Wutschitz, Lukas and Shokri, Reza and Brinton, Christopher G. and Sim, Robert},
  journal = {arXiv preprint arXiv:2506.04245},
  year    = {2025}
}

@article{chao2026stale,
  title   = {{STALE}: Can {LLM} Agents Know When Their Memories Are No Longer Valid?},
  author  = {Chao, Hanxiang and Bai, Yihan and Sheng, Rui and Li, Tianle and Sun, Yushi},
  journal = {arXiv preprint arXiv:2605.06527},
  year    = {2026}
}

@article{rein2023gpqa,
  title   = {{GPQA}: A Graduate-Level Google-Proof {Q\&A} Benchmark},
  author  = {Rein, David and Hou, Betty Li and Stickland, Asa Cooper and Petty, Jackson and Pang, Richard Yuanzhe and Dirani, Julien and Michael, Julian and Bowman, Samuel R.},
  journal = {arXiv preprint arXiv:2311.12022},
  year    = {2023}
}

@article{wang2024mmlupro,
  title   = {{MMLU-Pro}: A More Robust and Challenging Multi-Task Language Understanding Benchmark},
  author  = {Wang, Yubo and Ma, Xueguang and Zhang, Ge and Ni, Yuansheng and Chandra, Abhranil and Guo, Shiguang and Ren, Weiming and Arulraj, Aaran and He, Xuan and Jiang, Ziyan and Li, Tianle and Ku, Max and Wang, Kai and Zhuang, Alex and Fan, Rongqi and Yue, Xiang and Chen, Wenhu},
  journal = {arXiv preprint arXiv:2406.01574},
  year    = {2024}
}

@inproceedings{patil2025bfcl,
  title     = {The {Berkeley Function Calling Leaderboard (BFCL)}: From Tool Use to Agentic Evaluation of Large Language Models},
  author    = {Patil, Shishir G and Mao, Huanzhi and Yan, Fanjia and Ji, Charlie Cheng-Jie and Suresh, Vishnu and Stoica, Ion and Gonzalez, Joseph E.},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  pages     = {48371--48392},
  publisher = {PMLR},
  year      = {2025}
}

@article{patil2023gorilla,
  title   = {Gorilla: Large Language Model Connected with Massive {APIs}},
  author  = {Patil, Shishir G. and Zhang, Tianjun and Wang, Xin and Gonzalez, Joseph E.},
  journal = {arXiv preprint arXiv:2305.15334},
  year    = {2023}
}

@article{chhikara2025mem0,
  title   = {{Mem0}: Building Production-Ready {AI} Agents with Scalable Long-Term Memory},
  author  = {Chhikara, Prateek and Khant, Dev and Aryan, Saket and Singh, Taranjeet and Yadav, Deshraj},
  journal = {arXiv preprint arXiv:2504.19413},
  year    = {2025}
}

@article{packer2023memgpt,
  title   = {{MemGPT}: Towards {LLMs} as Operating Systems},
  author  = {Packer, Charles and Wooders, Sarah and Lin, Kevin and Fang, Vivian and Patil, Shishir G. and Stoica, Ion and Gonzalez, Joseph E.},
  journal = {arXiv preprint arXiv:2310.08560},
  year    = {2023}
}

@inproceedings{gutierrez2024hipporag,
  title     = {{HippoRAG}: Neurobiologically Inspired Long-Term Memory for Large Language Models},
  author    = {Guti{\'e}rrez, Bernal Jim{\'e}nez and Shu, Yiheng and Gu, Yu and Yasunaga, Michihiro and Su, Yu},
  booktitle = {The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year      = {2024}
}

@article{gutierrez2025hipporag2,
  title   = {From {RAG} to Memory: Non-Parametric Continual Learning for Large Language Models},
  author  = {Guti{\'e}rrez, Bernal Jim{\'e}nez and Shu, Yiheng and Qi, Weijian and Zhou, Sizhe and Su, Yu},
  journal = {arXiv preprint arXiv:2502.14802},
  year    = {2025}
}
```

MEME's paper is under anonymous review, so cite the
[dataset](https://huggingface.co/datasets/meme-benchmark/MEME) until it is
published.
