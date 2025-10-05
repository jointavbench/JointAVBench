# JointAVBench: A Benchmark for Joint Audio-Visual Reasoning Evaluation

<div style='display:flex; gap: 0.25rem; '>
<a href='/doc/JointAVBench.pdf'><img src='https://img.shields.io/badge/preprint-PDF-red'></a>
<a href='https://jointavbench.github.io'><img src='https://img.shields.io/badge/project-page-purple'></a>
<a href='https://huggingface.co/BrianatCambridge/video-SALMONN-o1'><img src='https://img.shields.io/badge/huggingface-benchmark-yellow'></a>
</div>

----

## TL;DR
This paper introduces a new method for generating audio-visual questions that requires both visual and auditory information to answer for evaluating omni-modal large language models (LLM). We also evaluate mainstream omni-LLMs on the benchmark and reveal barriers on omni-LLMs' development.

**Abstract**:

Compared to vision or audio large language models (LLMs), the key advantage of omni large language model, lies in their joint audio-visual reasoning capability. To train such models, datasets with questions requiring both visual and auditory information to answer are needed to build video understanding models. Moreover, videos contain complex audio signal types and scenes, interleaved with each other, demanding models with various cognitive capabilities. However, current datasets lack challenging multi-scene tasks, various types of audio information and cognition abilities. This paper introduces a dataset, called JointAVBench, designed to answer questions that necessitate AV integration, spanning 5 cognitive dimensions, 4 audio information types (speech, sound events, music, vocal traits), and 3 scene spans (single-, cross-, and full-scene). As manually crafting such questions is expensive and time-consuming, we propose leveraging state-of-the-art vision-LLMs, audio-LLMs and LLMs to automatically synthesize questions and answers that strictly require joint reasoning between visual and audio. Based on our JointAVBench dataset, we evaluate leading vision, audio, and omni LLMs. Results indicate that the top omni-LLM outperforms the top vision or audio LLMs but achieves only 56.2\% average accuracy, highlighting significant room for improvement, particularly in cross-scene reasoning.
