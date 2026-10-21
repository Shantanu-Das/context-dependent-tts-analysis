# Results

All numbers come from the executed `project.ipynb`; the files in `results/` hold the raw data behind them.

## Setup

- **Model and voice:** SpeechT5 (`microsoft/speecht5_tts`, character input) with the HiFi-GAN vocoder and the CMU ARCTIC `slt` voice.
- **Data:** 40 heteronym sentence pairs (10 words × 4 pairs), plus 16 control sentences. Each sentence is generated with 3 seeds.
- **Labels:** come from a phoneme recogniser (`wav2vec2-lv-60-espeak-cv-ft`), read inside the forced-aligned window of the target word and matched to the nearer CMUdict reference.

## Experiment 1: does context set the pronunciation?

**Evaluator check:** 48/48 control samples correct.

**Overall accuracy:**
- 146/240 samples = 60.8% (Wilson 95% CI 55–67%; this interval is too narrow because the 3 seeds of a sentence are correlated);
- 49/80 sentences by majority of seeds = 61% (95% CI 50–71%);
- 21 of the 240 samples are `unclear` (neither reading). All 3 seeds agree for 68 of 80 sentences.

| word | reading A | reading B | acc. context A | acc. context B | context effect | pairs: both correct / same reading both / other |
|---|---|---|---|---|---|---|
| read  | /ɹiːd/ present | /ɹɛd/ past | 0.25 | 0.92 | 0.25 | 1 / 3 / 0 |
| lead  | /liːd/ guide | /lɛd/ metal | 0.75 | 0.58 | 0.50 | 2 / 0 / 2 |
| live  | /lɪv/ verb | /laɪv/ adj | 1.00 | 0.00 | 0.33 | 0 / 3 / 1 |
| wind  | /wɪnd/ air | /waɪnd/ turn | 1.00 | 0.00 | 0.00 | 0 / 4 / 0 |
| tear  | /tɪɹ/ eye | /tɛɹ/ rip | 1.00 | 0.08 | 0.50 | 0 / 2 / 2 |
| bow   | /baʊ/ bend | /boʊ/ ribbon | 1.00 | 0.25 | 0.33 | 1 / 3 / 0 |
| bass  | /beɪs/ music | /bæs/ fish | 0.00 | 1.00 | 0.00 | 0 / 4 / 0 |
| wound | /wuːnd/ injury | /waʊnd/ wound up | 0.75 | 0.75 | 0.67 | 2 / 1 / 1 |
| close | /kloʊz/ shut | /kloʊs/ near | 0.83 | 1.00 | 0.83 | 3 / 0 / 1 |
| dove  | /dʌv/ bird | /doʊv/ dived | 1.00 | 0.00 | 0.08 | 0 / 4 / 0 |
| **total** | | | | | mean 0.35 | **9 / 24 / 7** |

How to read the table:
- **Context effect** = P(model says A | context A) − P(model says A | context B). It is 1 when the pronunciation follows the context perfectly and 0 when the context makes no difference.
- **"Same reading both"** means the model said the same reading in both sentences of the pair; 16 pairs used reading A for both, 8 used B for both.
- No pair was reversed.

**Interpretation.**
- The model does slightly better than a context-blind model, which would score 50%. Its dominant behaviour, though, is a **fixed default reading per word**, and the default can be either reading (*read* → /ɹɛd/, *bass* → /bæs/, *wind* → /wɪnd/).
- Only *close*, *wound* and partly *lead* respond to context.
- Several `unclear` cases are third pronunciations, such as /leɪd/ for *lead*, /lɛv/ for *live* and /tɜː/ for *tear*. These partly explain why *live*, *tear* and *bow* have a context effect above 0 even though they almost never produce reading B.
- One `unclear` case is an evaluator artefact: *close soon*, where the window catches the /s/ of *soon*.

## Experiment 2: is the intended reading decodable from the encoder?

The probe is a nearest-centroid classifier on the mean encoder state over the target word's characters. It is trained per heteronym, with leave-one-pair-out splits: each fold has 6 training sentences, and the 80 held-out predictions are spread across 40 folds per layer.

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | .59 | .60 | .60 | .60 | .64 | .60 | .61 | .61 | .65 | .69 | .71 | .69 | **.74** |

- **Layer 12** scores 59/80 = 74% (95% CI 63–82%). On the same sentences it beats layer 0 in 20 cases and loses in 8 (exact McNemar p = 0.036, approximate because the two sentences in a fold share a probe).
- **Layer 0** is non-contextual. It scores above 50% only because 24 of 40 pairs put the target at different character positions.
- **Probe vs speech:** at layer 12 the probe is right for 23/31 sentences the model mispronounced and 36/49 it pronounced correctly. For *read* (7/8) and *wind* (8/8) the probe separates the contexts even though the speech barely changes.

**Interpretation.** Information about the intended reading increasingly reaches the target word's encoder states in later layers, yet the spoken output often ignores it. This is a statement about decodability. The probe labels are confounded with the context words, the training sets are tiny, and none of this shows that the decoder uses this information.

## Hypotheses vs findings

| Hypothesis | Verdict |
|---|---|
| H1: context-appropriate pronunciation above 50% | Weakly supported (61%; sentence-level CI touches 50%) |
| H2: large differences between heteronyms | Supported (context effect 0.00–0.83) |
| H3: intended reading decodable in later layers above the non-contextual layer | Weakly supported (74% vs 59%, p ≈ 0.04) |
| Defaults come from training-data frequency | **Not tested.** This is a hypothesis only |

## Omitted or changed relative to the plan

- **No-context condition (single word):** dropped, because SpeechT5 produced garbled speech for one-word inputs (notebook section 3d).
- **Human listening annotation:** not done, since the analysis environment could not play audio. The automatic evaluator was validated on controls instead, and six example clips are in `results/audio_examples/` for a listening spot-check.
