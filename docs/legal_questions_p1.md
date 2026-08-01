# P1 legal questions

The following are hard questions for **ЮРИСТ**. Until answered, the named
decision stays blocked.

1. **Banking identifiers and anonymisation.** Are synthetic examples of card,
   account, BIK, INN, KPP, OGRN, and SNILS formats permissible in a published
   benchmark when all values are generated and cannot identify a person? What
   evidence of generation/anonymisation must be retained?
   **Blocks:** publication of RU-Critical v2 banking examples and any listener
   study material containing identifier-shaped strings.

2. **NonCommercial corpora.** Does research training on CC BY-NC,
   CC BY-NC-SA, or CC BY-NC-ND audio permit publication of checkpoints,
   embeddings, acoustic tokens, and generated examples? Is a trained model a
   derivative work for these sources?
   **Blocks:** training or checkpoint release from RUSLAN, OpenSTT, ESpeech
   Podcasts, and Balalaika beyond the current internal research use.

3. **ShareAlike propagation.** If RUSLAN or another ShareAlike source is mixed
   with CC0/CC BY data, which artifacts inherit ShareAlike: manifest, cached
   mels, tokenizer codes, model weights, or only redistributed audio?
   **Blocks:** licensing of a mixed-corpus Track N checkpoint and cache
   redistribution.

4. **Simulated telephony as a derivative.** Does G.711/Opus narrow-band
   simulation of third-party audio create a redistributable derived recording,
   and may hashes/metrics be published when the transformed audio may not?
   **Blocks:** redistribution of D12 audio and use of simulated telephony from
   any corpus whose license is not CC0/public domain.

5. **Russian personal-data rules and pseudonymous speaker IDs.** Under
   243-FZ and related requirements, are `client_id`, demographics, voice
   recordings, and speaker embeddings personal data, and must the processing
   infrastructure or consent evidence be located in Russia?
   **Blocks:** ingesting Common Voice demographic fields, persistent
   cross-corpus speaker clustering, and release of speaker-conditioned models.

6. **YouTube and source deletion obligations.** For YODAS/OpenSTT-derived
   material, what procedure is required when an upstream video is removed or
   its license changes? Is a pinned Hub revision sufficient provenance?
   **Blocks:** using YODAS/OpenSTT/Balalaika in long-lived training sets until a
   takedown and lineage procedure exists.

7. **Conflicting dataset declarations.** Which declaration controls when Hub
   metadata, README text, and the original source disagree (Golos,
   M-AILABS, ESpeech Webinars 2)?
   **Blocks:** all training and redistribution from those entries until the
   controlling rights holder and terms are documented.
