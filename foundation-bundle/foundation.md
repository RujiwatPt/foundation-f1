# Foundation Models

A **foundation model** is a large AI model trained on broad, general data so it can later be adapted to many different tasks.

Instead of training one model only for one narrow job, such as classifying cats or detecting spam, a foundation model is trained broadly first. After that, it can be adapted for tasks such as summarization, translation, coding, document question answering, image understanding, speech processing, OCR, search, and agent workflows.

---

## 1. What Makes It a Foundation Model?

A normal machine learning model is usually narrow.

Example:

```text
Input: X-ray image
Output: Disease label
```

A foundation model is broader.

Example:

```text
Input: Text, image, audio, video, code, or document
Output: Text, image, action, embedding, reasoning, or tool call
```

Key traits:

- **Large-scale data**: trained on huge datasets such as web pages, books, code, images, audio, and documents.
- **Self-supervised learning**: learns from raw data without needing humans to label every example.
- **Transferability**: one base model can be adapted to many downstream tasks.
- **General capability**: may support reasoning, writing, coding, translation, summarization, and multimodal understanding.

---

## 2. How a Foundation Model Is Built

A simplified pipeline looks like this:

```text
Data collection
→ Data cleaning and filtering
→ Tokenizer or representation design
→ Pretraining
→ Supervised fine-tuning
→ Preference tuning / alignment
→ Safety tuning
→ Evaluation
→ Red teaming
→ Staged release
→ Monitoring after release
```

---

## 3. Data Collection

The first step is gathering a very large and diverse dataset.

For a language model, data may include:

```text
Web pages
Books
Code
Academic text
Documents
Dialogue data
Licensed data
Synthetic data
Human-written examples
```

For multimodal models, data may include:

```text
Image-text pairs
Audio-text pairs
Video-text pairs
Screenshots
Documents
OCR data
Speech transcripts
```

The goal is not just memorization. The goal is to learn patterns: grammar, facts, reasoning structures, visual concepts, code patterns, and relationships between ideas.

---

## 4. Data Cleaning and Filtering

Raw internet-scale data is messy, so teams clean it before training.

They may remove or reduce:

```text
Duplicates
Spam
Broken text
Low-quality pages
Malware-related content
Private data
Toxic content
Personally identifiable information
Unlicensed or copyright-sensitive data, depending on policy
```

This step matters because bad training data can lead to bias, hallucination, unsafe advice, privacy leakage, or weak reasoning.

---

## 5. Tokenization and Representation

For text models, text is broken into small units called **tokens**.

Example:

```text
"I love machine learning"
```

may become something like:

```text
["I", " love", " machine", " learn", "ing"]
```

Most modern models use **subword** tokenization (such as BPE), so a single word can split into multiple tokens — here "learning" becomes "learn" + "ing". This lets the model handle rare words, typos, and new words by composing them from smaller pieces.

The model processes token IDs and learns patterns between them.

For images, the model may process pixels, patches, visual embeddings, or latent representations.

For audio, it may process waveform chunks, spectrogram-like features, or audio embeddings.

---

## 6. Pretraining

Pretraining is the expensive core training phase.

For large language models, the common objective is:

```text
Given previous tokens, predict the next token.
```

Example:

```text
Input:  The capital of Thailand is
Target: Bangkok
```

The model updates its parameters to reduce prediction error. Modern frontier models typically have hundreds of billions of parameters. Some mixture-of-experts (MoE) models report totals in the trillions, but only a fraction of those parameters are active for any given token.

This usually requires large GPU or TPU clusters.

---

## 7. Fine-Tuning

After pretraining, the base model may be knowledgeable but not yet useful as an assistant.

It may simply continue text instead of following instructions.

So teams fine-tune it on examples like:

```text
User: Explain RAG to a beginner.
Assistant: RAG stands for Retrieval-Augmented Generation...
```

Common fine-tuning types:

- **Supervised fine-tuning**: humans write ideal answers.
- **Instruction tuning**: the model learns to follow many kinds of instructions.
- **Domain tuning**: the model is specialized for fields like code, medicine, law, Thai language, OCR, finance, or customer support.
- **Multimodal tuning**: the model learns to connect images, audio, video, and text.

---

## 8. Preference Tuning and Alignment

The model is then trained to prefer better answers.

A simplified flow:

```text
The model generates multiple answers
→ Humans or AI judges rank the answers
→ A reward or preference model is trained
→ The assistant is optimized to produce preferred answers
```

This can improve:

```text
Helpfulness
Politeness
Instruction following
Refusal behavior
Safety
Formatting
Reasoning quality
```

These methods fall into two broad families:

- **Reward-model approaches**: a separate reward (preference) model is trained from rankings, then the assistant is optimized against it — usually with PPO. RLHF (human rankings) and RLAIF (AI rankings) work this way.
- **Direct preference approaches**: the assistant is optimized straight from the preference data with no separate reward model. DPO is the best-known example.

So PPO is the optimization algorithm used *inside* RLHF, not a separate alternative to it; DPO is the actual alternative that skips the reward model.

---

## 9. Safety Tuning

The model is tested and trained against dangerous or unwanted behavior.

Safety areas may include:

```text
Malware creation
Fraud
Privacy leakage
Medical overconfidence
Legal overconfidence
Self-harm encouragement
Biological or chemical misuse
Hate and harassment
Sexual content involving minors
Weapons guidance
Prompt injection vulnerability
```

Safety tuning does not make the model perfect, but it reduces risk before release.

---

## 10. How Foundation Models Are Evaluated Before Release

Evaluation is not one test. It is a full release process.

A serious foundation model is usually evaluated across:

```text
Capability
Robustness
Safety
Factuality
Fairness
Privacy
Real-world usability
```

---

## 11. Capability Evaluation

Teams test whether the model is actually useful.

Common areas:

```text
Reasoning
Math
Coding
Writing
Summarization
Translation
Instruction following
Tool use
Vision understanding
Audio understanding
Long-context recall
Document question answering
Multilingual ability
Domain knowledge
```

Benchmarks are useful, but they are not enough. A model can score well on a benchmark and still fail in real workflows.

---

## 12. Robustness Evaluation

Robustness testing checks whether the model works under messy conditions.

Examples:

```text
Typos
Ambiguous prompts
Long context
Contradictory instructions
Low-quality images
Tables
Scanned PDFs
Multi-step tasks
Weird formatting
Adversarial prompts
```

A model may work well on clean inputs but fail when the input is noisy, mixed-language, or poorly structured.

---

## 13. Safety Evaluation

Safety evaluation asks:

> Can the model help users do harmful things?

Common safety test areas:

```text
Cybersecurity misuse
Biosecurity misuse
Chemical misuse
Self-harm
Extremism
Fraud
Privacy leakage
Manipulation and persuasion
Child safety
Medical and legal risk
```

This becomes more important when the model can use tools, browse websites, read files, send emails, run code, or call APIs.

---

## 14. Hallucination and Factuality Evaluation

Foundation models can sound confident even when they are wrong.

Teams test:

```text
Closed-book factual question answering
Retrieval-grounded question answering
Citation accuracy
Summarization faithfulness
Contradiction detection
Refusal when uncertain
```

For RAG systems, this is critical. A model should not claim that a document says something unless that information is actually in the retrieved source.

It is also worth separating two distinct failure modes in RAG: the model can be unfaithful to a correct source, or the retrieval step can surface the wrong or incomplete sources in the first place. A faithful model given bad context will still produce a wrong answer, so retrieval quality (chunking, ranking, recall) must be evaluated on its own, not just the model's faithfulness.

---

## 15. Bias and Fairness Evaluation

Teams test whether the model behaves unfairly across different groups.

Examples:

```text
Gender bias
Racial or ethnic bias
Religious bias
Nationality bias
Language bias
Accent bias
Disability bias
Socioeconomic bias
```

This is difficult because bias is context-dependent. Passing one fairness benchmark does not guarantee fair behavior everywhere.

---

## 16. Privacy and Memorization Evaluation

Large models may accidentally memorize rare examples from training data.

Teams test:

```text
Can the model repeat private data?
Can it reveal emails, phone numbers, credentials, or private documents?
Can prompt attacks extract training data?
Did the model overfit duplicated data?
```

This is one reason data deduplication, privacy filtering, and access control are important.

---

## 17. Red Teaming

Red teaming means experts intentionally try to make the model fail.

They may test:

```text
Jailbreaks
Prompt injection
Dangerous instructions
Emotional manipulation
Scams
Policy bypasses
Hidden instructions
Model deception
Tool misuse
Data exfiltration
```

This is different from normal benchmark testing because humans actively search for unusual and dangerous failure modes.

---

## 18. System-Level Testing

The released product is not only the model.

A real AI product may include:

```text
System prompts
Safety classifiers
Retrieval systems
Tools
Browser access
Code execution
Image generation
Memory
Rate limits
Logging
Moderation layers
UI warnings
Human review processes
```

So teams must test the whole system, not only the raw model.

This is especially important for agents. A model that is safe in chat may become riskier when it can send emails, browse websites, run code, or modify files.

---

## 19. External Review and Staged Rollout

Before full release, companies may use:

```text
Internal testing
Dogfooding
Trusted testers
External red teamers
Limited beta
Enterprise pilots
Gradual rollout
Post-release monitoring
```

The purpose is to catch problems before the model reaches many users.

---

## 20. What “Release-Ready” Usually Means

A model is rarely perfect.

Release-ready usually means:

```text
Capability is useful enough
Known risks are below the release threshold
Safety mitigations are in place
Dangerous capabilities have been evaluated
Documentation exists
Monitoring is ready
Rollback plan exists
Abuse detection exists
```

A serious release may include:

```text
Model card
System card
Safety report
Risk assessment
Benchmark report
Known limitations
Intended use policy
Deployment guidelines
```

---

## 21. Choosing or Designing a Benchmark

A benchmark is only useful if it measures *your* actual workload. A model that tops a general knowledge or coding leaderboard may still fail on your real task. Before comparing scores, define the task, the inputs, and the success criteria that matter for your system — then judge benchmarks by how closely they match that.

Key things to check before trusting a benchmark:

- **Construct validity**: does the benchmark actually measure the capability you care about, or a proxy that happens to be easy to score?
- **Contamination**: the model may have seen the test set during training, which inflates scores. Compare the benchmark's release date against the model's training cutoff. Private or freshly held-out sets are more trustworthy.
- **Saturation**: if every candidate scores above ~90%, the benchmark can no longer separate them and is useless for your decision, even if it is an industry standard.
- **Metric and judge bias**: exact-match, F1, BLEU/ROUGE, pass@k, and LLM-as-judge each measure different things and fail differently. LLM judges in particular show position bias, verbosity bias, and self-preference, so do not treat one as ground truth without spot-checking.
- **Variance and significance**: a 1–2 point gap is often noise from sampling, seeds, or temperature. Run multiple times and look at confidence intervals before declaring a winner.
- **Harness sensitivity**: the same model scores very differently with different prompts, few-shot counts, and answer-parsing logic. When comparing models, the harness must be identical or the numbers are not comparable.
- **Cost and latency**: score quality together with tokens, cost per task, and latency. You are choosing a point on a quality-versus-cost frontier, not a single best model.
- **Distribution match**: a benchmark in one language or domain may not predict behavior on your data (other languages, specialized formats, unusual date systems, domain jargon).

Two points specific to agents:

- **Evaluate the trajectory, not just the final answer**: for tool-using agents, score the intermediate steps, tool calls, and error recovery. A correct final answer reached by luck is not robust.
- **Treat adversarial robustness as a separate eval**: prompt-injection and jailbreak resistance need their own attack set and their own metric (for example, successful-injection rate). Accuracy benchmarks will not surface these failures at all.

Practical takeaway: no public benchmark will match a specialized domain exactly. Build a small golden set — a few dozen to a few hundred real, labeled examples from your own data — and treat that as your primary benchmark, with public ones as secondary signal.

---

## 22. Simple Analogy

Training and releasing a foundation model is like training a new employee.

- **Pretraining** is like letting them read a huge library.
- **Fine-tuning** is like teaching them company procedures.
- **Preference tuning** is like showing them what good answers look like.
- **Safety tuning** is like teaching them what not to do.
- **Evaluation** is like exams, interviews, audits, and trial work.
- **Red teaming** is like hiring people to trick them and see where they fail.
- **Staged release** is like letting them first help internally, then a few trusted customers, then everyone.

---

## 23. Important Limitations

Foundation models are powerful, but they are not databases and not perfectly reliable by default.

They can:

```text
Hallucinate
Overgeneralize
Reflect bias in training data
Fail on rare cases
Sound confident when wrong
Be vulnerable to prompt injection
Misuse tools
Leak information if badly designed
```

For production use, especially in RAG, healthcare, finance, legal, security, or automation, the foundation model should be wrapped with:

```text
Retrieval grounding
Access control
Logging
Human review
Evaluation sets
Guardrails
Rate limits
Permission boundaries
Fallback behavior
```

---

## 24. Practical Takeaway

The foundation model is the reasoning and language engine.

But a real AI product still needs:

```text
Data control
Evaluation
Safety gates
Workflow design
Monitoring
Human oversight
Security boundaries
```

For RAG and agent systems, do not think only about the model. Think about the full system around the model.
