# CV + LLM 16-Week Upskilling Sprint

Learning plan blending modern computer vision with LLM systems. One demo or service every week. This repo is your single source of truth for goals, progress, and results.

---

## Quick start

1. **Clone** this repo and open the **Project board**
   - Project: [cv-llm-upskilling-16w](https://github.com/phemiloyeAI/cv-llm-upskilling-16w)
   - Scoreboard: see [SCOREBOARD.md](./SCOREBOARD.md)
2. **Kick off the week**
   - Run the workflow: **Actions -> Weekly kickoff -> Run workflow**
   - This creates the "Week NN" issue and updates the scoreboard
3. **Work in public**
   - Use the issue template **Weekly Progress (Public)** under **New issue**
   - Post a short summary on LinkedIn/Twitter and paste links in the issue

---

## Weekly cadence

- **Plan (Mon)** set a numerical target and list deliverables
- **Build (Tue-Thu)** code, measure, document
- **Publish (Fri-Sun)** open PRs, write short notes, post publicly, close the issue

**Minimum each week**
- [ ] Reproduce one figure or result from a paper/repo
- [ ] Run one ablation or micro-benchmark
- [ ] Update a demo/service and commit a perf sheet (latency, accuracy, cost)

---

## Hard-mode acceptance criteria

- [ ] Numerical target defined and met for the week
- [ ] At least one concrete artifact shipped (PR, demo, service, or report)
- [ ] Perf and unit tests pass in CI
- [ ] "What sped up, what broke, what's next" logged in the issue

---

## 16-week plan at a glance

### Month 1 - Training efficiency and ablations
**Week 01 - PyTorch 2.x compile + profiling**
- Targets: >=1.5x speedup on at least one model, >=20% on all three, identical metrics
- Deliverables:
  - [ ] Benchmarks on ResNet50, ViT-B, UNet with `torch.compile`, CUDA graphs, `torch.profiler`
  - [ ] Trace exports and short write-up

**Week 02 - Attention hot path: FlashAttention**
- Targets: >=1.5x attention kernel speedup, <=1% task metric delta
- Deliverables:
  - [ ] ViT or tiny-decoder with FlashAttention toggle (FP16/FP8 if available)
  - [ ] Ablation table and parity checks

**Week 03 - Data engine throughput**
- Targets: dataloader never the bottleneck, >=90% GPU util during train steps
- Deliverables:
  - [ ] WebDataset shards, async prefetch, GPU JPEG decode
  - [ ] Duplicate removal via MinHash and input-idle time dashboard

**Week 04 - Finetuning toolbelt**
- Targets: +1.0-1.5% top-1 or +mAP@50 at <=1.1x cost
- Deliverables:
  - [ ] Baseline vs LoRA/QLoRA, AMP, EMA, mixup/cutmix, grad-ckpt
  - [ ] Hydra-configured template with tests

### Month 2 - Modern CV + deploy
**Week 05 - SSL features as a service (DINOv2)**
- Targets: linear probe >=5-10% over scratch with 10x fewer labels
- Deliverables:
  - [ ] Feature service (REST/gRPC) + feature bank
  - [ ] Retrieval demo and probe results

**Week 06 - Real-time detection that ships (YOLOv10 + TensorRT/Triton)**
- Targets: GPU p95 <12 ms, CPU p95 <60 ms; mAP within 1-2 points of PyTorch
- Deliverables:
  - [ ] ONNX -> TensorRT export and Triton model repo
  - [ ] Perf sheet and profiling traces

**Week 07 - Promptable segmentation (SAM 2)**
- Targets: 2-3 clicks to 85% IoU; 720p video real-time
- Deliverables:
  - [ ] Web demo for image + video segmentation
  - [ ] Clicks-to-IoU study vs baseline

**Week 08 - Multi-object tracking + ReID**
- Targets: IDF1 >=0.70 on small eval; <=2% ID switches/frame
- Deliverables:
  - [ ] ByteTrack/BoT-SORT with SSL appearance features
  - [ ] MOT metrics and qualitative videos

### Month 3 - Multimodal + serving
**Week 09 - VLM quickstart + domain LoRA**
- Targets: >=10% relative gain on domain eval vs base
- Deliverables:
  - [ ] Run open VLM (e.g. Qwen2-VL/LLaVA-OneVision), small domain LoRA
  - [ ] Notebook + error taxonomy

**Week 10 - High-throughput serving**
- Targets: >=2x throughput vs vanilla HF; p95 latency goal met
- Deliverables:
  - [ ] vLLM for text head; Triton/TensorRT for vision encoders
  - [ ] docker-compose "one-box" + Prometheus/Grafana

**Week 11 - Multimodal RAG + search**
- Targets: +10% nDCG@10 vs naive embeddings
- Deliverables:
  - [ ] Hybrid BM25+embeddings over images+text, cross-encoder reranker
  - [ ] Query analytics + A/B switch

**Week 12 - 3D from casual capture (3DGS)**
- Targets: interactive viewer >=60 fps; quality table (PSNR/LPIPS)
- Deliverables:
  - [ ] Capture-to-viewer pipeline; NeRF comparison
  - [ ] Simple web viewer

### Month 4 - Architecture, reliability, cost
**Week 13 - Positional methods: RoPE, scaled-RoPE/YaRN, ALiBi**
- Targets: 8k-context ppl within 5% of 2k baseline using scaling; exact decoding >2x train length
- Deliverables:
  - [ ] Toggleable positional modules + ablation report

**Week 14 - FFN and norm: SwiGLU + RMSNorm**
- Targets: <=same params/latency with equal or better perplexity; stable training
- Deliverables:
  - [ ] Drop-in modules with benchmarks and LR sensitivity study

**Week 15 - Inference: quant + decoding**
- Targets: >=1.7x tokens/s at <=1% quality delta; >=2x lower memory
- Deliverables:
  - [ ] AWQ/GPTQ weight-only quant; KV cache study; speculative decoding

**Week 16 - Reliability + cost and the honest perf report**
- Targets: SLOs met in 24h soak; cost per 1k requests documented
- Deliverables:
  - [ ] Blue-green deploy + autoscaling
  - [ ] Drift/robustness checks and final report

---

## Reading lists with checkboxes

> Mark items as you go. Add the links you use right next to each line.

### Core efficiency (Weeks 01-04)
| Topic | Must read / watch | Done |
|---|---|---|
| PyTorch 2.x compile + Inductor | [ ] Docs and deep-dive notes | [ ] |
| CUDA graphs + profiler | [ ] Official docs and tutorials | [ ] |
| FlashAttention (2/3) | [ ] Paper + repo readthrough | [ ] |
| Data pipelines at scale | [ ] WebDataset + GPU JPEG decode notes | [ ] |
| Finetuning tricks | [ ] LoRA/QLoRA, EMA, AMP, grad-ckpt | [ ] |

### Modern CV (Weeks 05-08)
| Topic | Must read / repo | Done |
|---|---|---|
| DINOv2 | [ ] Paper + model card | [ ] |
| YOLOv10 | [ ] Paper + training tips | [ ] |
| SAM 2 | [ ] Paper + video memory design | [ ] |
| Tracking | [ ] ByteTrack / BoT-SORT papers + code | [ ] |

### Multimodal + Serving (Weeks 09-12)
| Topic | Must read / repo | Done |
|---|---|---|
| VLMs (Qwen2-VL, LLaVA-OneVision, etc.) | [ ] Model cards + eval readme | [ ] |
| vLLM | [ ] Paged attention + prefix caching notes | [ ] |
| Multimodal RAG | [ ] Retrieval + reranking survey | [ ] |
| 3D Gaussian Splatting | [ ] Paper + viewer code | [ ] |

### Architectures + Inference (Weeks 13-16)
| Topic | Must read / repo | Done |
|---|---|---|
| Positional methods | [ ] RoPE, ALiBi, NTK/YaRN scaling | [ ] |
| FFN + Norms | [ ] SwiGLU, RMSNorm, PaLM section | [ ] |
| Quantization | [ ] AWQ, GPTQ papers and tools | [ ] |
| Decoding | [ ] Speculative decoding primers | [ ] |

---

## Weekly post template

> Copy to LinkedIn/Twitter and your Week issue

