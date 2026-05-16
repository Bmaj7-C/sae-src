# Urban Spatial Concepts in LLM Feature Space

Urban science increasingly explores the use of LLMs (e.g., synthetic travel survey generation, context completion for passive traces). Domain applications can invoke compound planning concepts such as transit-oriented development, whose components --transit, density, walkability-- encode spatial structure. While the composition is present in corpus, co-occurrence does not guarantee internal representation.

-  Do LLMs internalize urban spatial concepts, or just labels?
-  How this propagates to generated output, and is it fixable?

> **Sparse Autoencoder (SAE)** is a learned dictionary that decomposes a model's activation vector $\mathbf{x}$ into a sparse weighted sum of interpretable features $\mathbf{f}_i$, such that $\mathbf{x} \approx \sum_i a_i \mathbf{f}_i$, where most $a_i = 0$. Sparsity forces each feature to capture a distinct pattern, learned unsupervised from the model's own activations.


<br>

```mermaid
%%{init: {'theme': 'neutral'}}%%
flowchart LR
    M["Gemma 2 9B-IT<br/>GemmaScope<br/>Residual 131k<br/>Layers 9, 20, 31"]

    subgraph Discovery
        D["SAE Decompose"] --> P["Probe"]
    end

    subgraph Application
        S["Steer"] --> T["Travel Diary<br/>Generation"]
    end

    subgraph Evaluation
        E["Measure Behavioral<br/>Signature Shifts"]
    end

    M --> D
    P --> S
    T --> E
```

#### Implement
`config/` defines concept hierarchy, prompt variants, and steering conditions. `core/` wraps [Neuronpedia](http://neuronpedia.org/) APIs, probes target tokens across prompts, and computes Jaccard overlap and composition recovery. `experiments/` run phases end-to-end.

```mermaid
graph LR
    subgraph Probe ["<b>Probe</b>"]
        direction TB
        P1["10 primitives · \n3 composites\n2 synonym · 2 opposition\n5 prompts each"] -->|"SAE probe\ntop-k features\nat target token"| P2["Feature set\nper concept"]
        P2 --> P3["Stability\nJaccard across\nprompts"]
        P2 --> P4["Calibration\nJaccard across\nsynonym/opposition"]
        P2 --> P5["Composition recovery\nprimitive features found\nin composite / total"]
    end
    subgraph Steer ["<b>Steer</b>"]
        direction TB
        S1["Diary prompt\n+ rank-1 primitive feature\n6 conditions · n=50 each"] -->|"Neuronpedia steer\nstrength=20 · temp=0.5"| S2["Generated\ndiaries"] -->|"Qwen-3 8B\ntrip extraction"| S3["Eval:\nactive mode share\ndestination POI diversity"]
    end
    Probe ~~~ Steer
    style Probe fill:none,stroke:none
    style Steer fill:none,stroke:none
```

```
📁
├── config/
│   ├── concepts.yaml
│   └── extraction_prompt.txt
├── core/
│   ├── analysis.py
│   ├── client.py
│   └── probes.py
└── experiments/
    ├── composite.py
    ├── opposition.py
    ├── primitives.py
    ├── steering.py
    └── synonymy.py
```