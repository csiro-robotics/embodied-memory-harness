# Qwen3 tokenizer (pinned)

`tokenizer.json` is the tokenizer of Qwen/Qwen3-14B, the model the benchmark serves as
`qwen3:14b` in Ollama. `src/tokens.py` uses it to count the memory budget, the injection
meter and every token figure the harness computes itself (fix plan H6).

- Source: https://huggingface.co/Qwen/Qwen3-14B/blob/40c069824f4251a91eefaf281ebe4c544efd3e18/tokenizer.json
- Revision: `40c069824f4251a91eefaf281ebe4c544efd3e18` (downloaded 2026-09-27)
- SHA-256: `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`
- Licence: Apache 2.0, copied here as `LICENSE`.

It is committed so that compute nodes need no network access and every job counts tokens
with the same file. Set `QWEN3_TOKENIZER` to use a copy elsewhere.
