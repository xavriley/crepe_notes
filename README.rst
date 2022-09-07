===========
CREPE notes
===========


.. image:: https://img.shields.io/pypi/v/crepe_notes.svg
        :target: https://pypi.python.org/pypi/crepe_notes

.. image:: https://img.shields.io/travis/xavriley/crepe_notes.svg
        :target: https://travis-ci.com/xavriley/crepe_notes

.. image:: https://readthedocs.org/projects/crepe-notes/badge/?version=latest
        :target: https://crepe-notes.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Post-processing for CREPE to turn f0 pitch estimates into discrete notes e.g. MIDI


* Free software: MIT license
* Documentation: https://crepe-notes.readthedocs.io.

Usage
-----

```
crepe_notes [path_to_crepe_f0_file] [path_to_original_audio]
```

A '.mid' file will be created in the location of the CREPE f0 file with the same name.


Features
--------

* outputs midi from CREPE pitch estimates

Caveats
-------

Doesn't handle repeated notes well. This is also more likely to produce spurious extra notes in quiet sections, but these can be filtered by velocity in other midi programs.

Roadmap
-------

-[ ] Handle repeated notes better
-[ ] Allow filtering by minimum velocity/amplitude
-[ ] Allow filtering by minimum note length
-[ ] Optionally output activation graphs
-[ ] Add option to output CSV format
-[ ] (Distant goal) Add UI to aid with picking thresholds for velocity and note length
-[ ] Experiment with edge preserving smoothing on the confidence thresholds to reduce spurious grace notes/glissandi

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
