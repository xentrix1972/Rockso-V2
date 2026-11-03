# Public-release checklist

This preparation is for review. It does not change repository visibility, merge
the preparation branch, update Hugging Face, rewrite history or announce a release.

Completed in the prepared change:

- [x] Owner selected Apache-2.0 for original code/documentation; official license
  text, scope and notices included. No blanket weights/data grant.
- [x] Retain third-party MIT/Apache license texts and pinned source references,
  with verified correspondences and historical-provenance limits stated.
- [x] Link the public weights-only checkpoint at an immutable Hub revision and
  state its hash, architecture, counters' provenance and non-resumable status.
- [x] Supply and test CPU generation, native PrefixLM conditions, exact tokenizer
  identity and both publication/original tokenizer file layouts.
- [x] Keep training algorithms, runtime settings, launcher and architecture intact.
- [x] Complete CPU/static tests and bounded current-source/history screening;
  publish the validation scope without performance or benchmark claims.

Before the repository visibility change:

- [ ] Review and merge the preparation PR, or select the exact reviewed commit.
- [ ] Owner accepts the existing public-facing author identity/contact metadata
  and intentional original-run W&B link. No hidden-history purge is proposed.
- [ ] Review the remaining provenance limits: OLMo's historical import commit and
  every intermediate local adaptation are not reconstructed. The matching
  reviewed upstream files, named modded-nanogpt record, archived Violetto pin
  and harness fixture sources are recorded. Add any other known copied sources.
- [ ] Give the final visibility/publication instruction for the reviewed version.

Separate from a code-only release:

- [ ] Confirm Sapient's dataset/tokenizer redistribution and reuse terms with an
  authoritative license or permission. The sampled-data card reviewed here has
  no explicit license; the separate HRM-Text code license is not a substitute.
- [ ] Choose and document a weights license only after the owner resolves the
  relevant tokenizer/training-data provenance and permissions. This preparation
  does not license or modify the already-public Hugging Face artifacts.
- [ ] Any future GPU parity, speed or quality claims require separate tests with
  pinned checkpoint, data, prompt/condition and environment identities.
