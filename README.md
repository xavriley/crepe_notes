CREPE notes
===========


<!-- .. image:: https://img.shields.io/pypi/v/crepe_notes.svg -->
<!--         :target: https://pypi.python.org/pypi/crepe_notes -->

<!-- .. image:: https://img.shields.io/travis/xavriley/crepe_notes.svg -->
<!--         :target: https://travis-ci.com/xavriley/crepe_notes -->

<!-- .. image:: https://readthedocs.org/projects/crepe-notes/badge/?version=latest -->
<!--         :target: https://crepe-notes.readthedocs.io/en/latest/?version=latest -->
<!--         :alt: Documentation Status -->

Post-processing for the CREPE pitch tracker to turn f0 pitch estimates into discrete notes (MIDI)

https://github.com/xavriley/crepe_notes/assets/369527/4cc895f5-9bfe-47af-809d-0152933dc4c9

[Demo video for pypi users also available here](https://www.youtube.com/watch?v=vFvbedBgLKg)

Features
--------

* outputs midi notes for monophonic audio
* include MIDI velocity information
* includes options to filter notes that are too quiet or too short
* experimental support for alternative pitch tracking backends (PESTO, PENN, torchcrepe)

Installation
------------

```bash
pip install crepe-notes[crepe]
```

### Python 3.10+

The PyPI version of madmom has a known compatibility issue with Python 3.10+. After installing, replace it with the version from git:

```bash
pip install git+https://github.com/CPJKU/madmom@main
```

### macOS / Apple Silicon

The `crepe` package depends on `pkg_resources` at build time, which was removed from `setuptools>=82`. If you get a `ModuleNotFoundError: No module named 'pkg_resources'` error, install with:

```bash
pip install "setuptools<81"
pip install crepe --no-build-isolation
pip install crepe-notes[crepe]
```

### Experimental backends

The following alternative pitch trackers can be used instead of CREPE. These are **experimental** and have not been fully evaluated — results may differ from the published benchmarks below.

```bash
# PESTO (lightweight, PyTorch-based, works on Apple Silicon/MPS)
pip install crepe-notes[pesto]

# PENN (PyTorch-based, works on Apple Silicon/MPS)
pip install crepe-notes[penn]

# torchcrepe (PyTorch reimplementation of CREPE, works on Apple Silicon/MPS)
pip install crepe-notes[torchcrepe]
```

Basic Usage
-----------

```bash
# Using CREPE (default, recommended)
crepe_notes [path_to_original_audio]

# Using an experimental backend
crepe_notes --pitch-tracker pesto [path_to_original_audio]
crepe_notes --pitch-tracker penn [path_to_original_audio]
crepe_notes --pitch-tracker torchcrepe [path_to_original_audio]
```

A '.mid' file will be created in the current directory with the name `[audio_file_stem].[pitch_tracker].transcription.mid`.

For additional options check out `crepe_notes --help`.

## Min duration, min velocity and sensitivity

These are the three params you may need to tweak to get optimal results.

* `--min-duration` is specified in seconds (e.g. `0.03` is `30ms`). For fast, virtuosic music this is a reasonable default but for things like vocals and double bass lines a longer min duration (`50ms` or higher) may reduce the number of errors in your transcription.

* `--min-velocity` is expressed as in MIDI e.g. `0 - 127`. The default is `6` which removes any notes with velocities at or below that value, but you may find recordings with a higher noise floor benefit from a higher threshold.

* `--sensitivity` relates to the peak picking threshold used on the combined signal (see paper for details) and defaults to `0.001`. If the source material has an unstable pitch profile which results in a lot of short notes either side of a longer target note, increasing the sensitivity to `0.002` may help. 


## Caching data files

If you are running `crepe_notes` over an entire dataset, we recommend using the `--save-analysis-files` flag. This will write the following results:

* pitch tracker output to `[audio_file_stem].[pitch_tracker].f0.csv`
* madmom onset activations to `[audio_file_stem].onsets.npz`
* amplitude envelope calculations to `[audio_file_stem].amp_envelope.npz`

This will speed up run times at the expense of some disk space.

About
-----

This repo is the code to accompany the following paper:

> X. Riley and S. Dixon, “CREPE Notes: A new method for segmenting pitch contours into discrete notes,” in Proceedings of the 20th Sound and Music Computing Conference, Stockholm, Sweden, 2023, pp. 1–5.

In the paper we propose a method of combining two things:

a) the gradient of the pitch contour from CREPE
b) the (inverse) confidence of the pitch estimate from CREPE

This gives us a new signal which is a reliable indicator of note onsets, which we can then use to segment the pitch contour into discrete notes. For more details please see the paper or the demo video above.

Results
-------

How good is it? For the datasets we've tested so far it looks promising.

* FiloSax (24 hrs solo saxophone audio) - 90% F-measure (no offsets)
* ITM GT Flute 99 - (20mins Irish trad flute) - 74% F-measure (no offsets) +7% over Basic Pitch
* FiloBass (4 hrs double bass source separated stems) - 72% F-measure (no offsets) +10% over Basic Pitch

Please open a Github issue if you get results for any other public datasets - we'll try to include them in this repo.

Caveats
-------

All supported pitch trackers only work for monophonic audio, which means crepe_notes only works for monophonic audio too. If you need polyphonic transcription, check out [Basic Pitch](https://basicpitch.spotify.com/).

### Pitch tracker backends

| Backend | Package | GPU/MPS Support | Status |
|---------|---------|----------------|--------|
| [CREPE](https://github.com/marl/crepe) | `crepe_notes[crepe]` | TensorFlow GPU | **Recommended** — fully evaluated, used in published results |
| [PESTO](https://github.com/SonyCSLParis/pesto) | `crepe_notes[pesto]` | CUDA, MPS (Apple Silicon) | Experimental |
| [PENN](https://github.com/interactiveaudiolab/penn) | `crepe_notes[penn]` | CUDA, MPS (Apple Silicon) | Experimental |
| [torchcrepe](https://github.com/maxrmorrison/torchcrepe) | `crepe_notes[torchcrepe]` | CUDA, MPS (Apple Silicon) | Experimental |

Due to the way the algorithm works, repeated notes at the same pitch are treated as a special case and have to fall back to using a standard onset detector (madmom). The results might vary depending on the type of music you want to transcribe. For example, in a jazz saxophone solo it's relatively uncommon to repeat the same note. In a rock bass line however the opposite is true.

The onset detection library we use ([madmom](https://github.com/CPJKU/madmom)) has a licence which restricts commercial use. This restrictions is conferred onto CREPE Notes as a result - if you have a commercial use case please contact the madmom authors to discuss this.

Roadmap
-------

- [ ] (Distant goal) Add UI to aid with picking thresholds for velocity and note length
- [x] Experiment with edge preserving smoothing on the confidence thresholds to reduce spurious grace notes/glissandi

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

* _Cookiecutter_: https://github.com/audreyr/cookiecutter
* _`audreyr/cookiecutter-pypackage`_: https://github.com/audreyr/cookiecutter-pypackage
