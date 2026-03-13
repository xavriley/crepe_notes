#!/usr/bin/env python
"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [
    'Click>=8.0',
    'librosa>=0.10.0',
    'numpy>=2.0',
    'scipy>=1.13',
    'madmom',
    'pretty_midi',
    'tqdm',
]

extras_require = {
    'crepe': ['crepe', 'tensorflow'],
    'pesto': ['pesto-pitch', 'torchaudio'],
    'penn': ['penn', 'torchaudio'],
    'torchcrepe': ['torchcrepe', 'torchaudio'],
}

test_requirements = []

setup(
    author="Xavier Riley",
    author_email='xavriley@hotmail.com',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    description=
    "Post-processing for CREPE to turn f0 pitch estimates into discrete notes e.g. MIDI",
    entry_points={
        'console_scripts': [
            'crepe_notes=crepe_notes.cli:main',
        ],
    },
    install_requires=requirements,
    extras_require=extras_require,
    license="GNU General Public License v3.0 license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='crepe_notes',
    name='crepe_notes',
    packages=find_packages(exclude=['tests*'],include=['crepe_notes', 'crepe_notes.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/xavriley/crepe_notes',
    version='0.2.0',
    zip_safe=False,
)
