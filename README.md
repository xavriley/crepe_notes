CREPE notes
===========


<!-- .. image:: https://img.shields.io/pypi/v/crepe_notes.svg -->
<!--         :target: https://pypi.python.org/pypi/crepe_notes -->

<!-- .. image:: https://img.shields.io/travis/xavriley/crepe_notes.svg -->
<!--         :target: https://travis-ci.com/xavriley/crepe_notes -->

<!-- .. image:: https://readthedocs.org/projects/crepe-notes/badge/?version=latest -->
<!--         :target: https://crepe-notes.readthedocs.io/en/latest/?version=latest -->
<!--         :alt: Documentation Status -->

Post-processing for CREPE to turn f0 pitch estimates into discrete notes e.g. MIDI

[<img src="https://github.com/xavriley/crepe_notes/assets/369527/5873cfed-13e4-4837-b10a-c04c5243cb5e" width="50%">](https://www.youtube.com/watch?v=vFvbedBgLKg)


* Free software: mainly MIT licensed, some dependencies have restrictions on commercial use
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

This gives us a new signal which is a reliable indicator of note onsets, which we can then use to segment the pitch contour into discrete notes. For more details please see the paper.



Caveats
-------

Doesn't handle repeated notes well. This is also more likely to produce spurious extra notes in quiet sections, but these can be filtered by velocity in other midi programs.

Roadmap
-------

-[ ] (Distant goal) Add UI to aid with picking thresholds for velocity and note length
-[ ] Experiment with edge preserving smoothing on the confidence thresholds to reduce spurious grace notes/glissandi

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

* _Cookiecutter_: https://github.com/audreyr/cookiecutter
* _`audreyr/cookiecutter-pypackage`_: https://github.com/audreyr/cookiecutter-pypackage
