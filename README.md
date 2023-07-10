CREPE notes
===========


<!-- .. image:: https://img.shields.io/pypi/v/crepe_notes.svg -->
<!--         :target: https://pypi.python.org/pypi/crepe_notes -->

<!-- .. image:: https://img.shields.io/travis/xavriley/crepe_notes.svg -->
<!--         :target: https://travis-ci.com/xavriley/crepe_notes -->

<!-- .. image:: https://readthedocs.org/projects/crepe-notes/badge/?version=latest -->
<!--         :target: https://crepe-notes.readthedocs.io/en/latest/?version=latest -->
<!--         :alt: Documentation Status -->

Post-processing for CREPE to turn f0 pitch estimates into discrete notes (MIDI)

https://github.com/xavriley/crepe_notes/assets/369527/4cc895f5-9bfe-47af-809d-0152933dc4c9


* Free software: mainly MIT licensed, some dependencies (madmom) have restrictions on commercial use
* Documentation: https://crepe-notes.readthedocs.io.
  
Installation
------------

```
pip install crepe_notes
```

Basic Usage
-----------

```
crepe_notes [path_to_original_audio]
```

A '.mid' file will be created in the location of the audio file with the name `[audio_file_stem].transcription.mid`.

For additional options check out `crepe_notes --help`.

Features
--------

* outputs midi notes for monophonic audio
* include MIDI velocity information
* includes options to filter notes that are too quiet or too short


About
-----

This repo is the code to accompany the following paper:

> X. Riley and S. Dixon, “CREPE Notes: A new method for segmenting pitch contours into discrete notes,” in Proceedings of the 20th Sound and Music Computing Conference, Stockholm, Sweden, 2023, pp. 1–5.

In the paper we propose a method of combining two things:

a) the gradient of the pitch contour from CREPE
b) the (inverse) confidence of the pitch estimate from CREPE

This gives us a new signal which is a reliable indicator of note onsets, which we can then use to segment the pitch contour into discrete notes. For more details please see the paper or the demo video above.

Caveats
-------

CREPE only works for monophonic audio, which means CREPE Notes only works for monophonic audio too. If you need polyphonic transcription, check out [Basic Pitch](https://basicpitch.spotify.com/).

Due to the way the algorithm works, repeated notes at the same pitch are treated as a special case and have to fall back to using a standard onset detector (madmom). The results might vary depending on the type of music you want to transcribe. For example, in a jazz saxophone solo it's relatively uncommon to repeat the same note. In a rock bass line however the opposite is true.

Roadmap
-------

-[ ] (Distant goal) Add UI to aid with picking thresholds for velocity and note length
-[x] Experiment with edge preserving smoothing on the confidence thresholds to reduce spurious grace notes/glissandi

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

* _Cookiecutter_: https://github.com/audreyr/cookiecutter
* _`audreyr/cookiecutter-pypackage`_: https://github.com/audreyr/cookiecutter-pypackage
