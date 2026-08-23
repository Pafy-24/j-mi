# Surse tehnice și atribuiri

Implementarea MI este originală în acest repository, dar designul block expansion a fost construit după următoarele surse primare și exemple open-source:

1. Zhang et al., **LLaMA Pro: Progressive LLaMA with Block Expansion**, arXiv:2401.02415 / ACL 2024 — https://arxiv.org/abs/2401.02415
2. Implementarea oficială TencentARC/LLaMA-Pro, commit inspectat `bead65718f9d2c7877a170f953ead24d2cc53b91`, Apache-2.0 — https://github.com/TencentARC/LLaMA-Pro
3. Implementarea AnswerDotAI/fsdp_qlora, commit inspectat `05ed9f2a60f96a0795cb082bceab70a9b19fd213`, Apache-2.0 — https://github.com/AnswerDotAI/fsdp_qlora
4. Documentația oficială Hugging Face pentru cache și chat templates — https://huggingface.co/docs/transformers/main/en/kv_cache și https://huggingface.co/docs/transformers/main/en/chat_templating

Elementele preluate conceptual sunt copierea și inserarea blocurilor, inițializarea identitară prin zeroizarea proiecțiilor reziduale de ieșire, înghețarea parametrilor originali și salvarea separată a parametrilor extensiei. Codul sursă din repository-urile de mai sus nu a fost copiat textual în MI.
