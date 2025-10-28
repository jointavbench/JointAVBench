# JointAVBench: A Benchmark for Joint Audio-Visual Reasoning Evaluation

<div style='display:flex; gap: 0.25rem; '>
<a href='/doc/JointAVBench.pdf'><img src='https://img.shields.io/badge/preprint-PDF-red'></a>
<a href='https://jointavbench.github.io'><img src='https://img.shields.io/badge/project-page-purple'></a>
<a href='https://huggingface.co/datasets/JointAVBench/JointAVBench'><img src='https://img.shields.io/badge/huggingface-benchmark-yellow'></a>
</div>

----

## TL;DR
This paper introduces a new method for generating audio-visual questions that requires both visual and auditory information to answer for evaluating omni-modal large language models (LLM). We also evaluate mainstream omni-LLMs on the benchmark and reveal barriers on omni-LLMs' development.

**Abstract**:

Understanding videos inherently requires reasoning over both visual and auditory information. To properly evaluate Omni-Large Language Models (Omni-LLMs), which are capable of processing multi-modal information including vision and audio, an effective benchmark must comprehensively cover three key aspects: (1) multi-modal dependency (i.e., questions that cannot be answered using vision or audio alone), (2) diverse audio information types (e.g., speech, sound events), and (3) varying scene spans. However, existing datasets fall short in one or more of these dimensions, limiting strict and comprehensive evaluation. To address this gap, we introduce JointAVBench, a novel benchmark with strict audio-video correlation, spanning five cognitive dimensions, four audio information types (speech, sound events, music, vocal traits), and three scene spans (single-, cross-, and full-scene). Given the high cost of manual annotation, we propose an automated pipeline that leverages state-of-the-art vision-LLMs, audio-LLMs, and general-purpose LLMs to synthesize questions and answers that strictly require joint audio-visual understanding.  We evaluate leading vision-only, audio-only, and Omni-LLMs on our dataset. Results show that even the best-performing Omni-LLM achieves only 56.2\% average accuracy, outperforming uni-modal baselines but revealing substantial room for improvement, especially in cross-scene reasoning.
