#!/usr/bin/env python
"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', 'librosa>=0.7.2', 'numpy>=1.20.3', 'madmom', 'pretty_midi']

test_requirements = []

setup(
    author="Xavier Riley",
    author_email='xavriley@hotmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description=
    "Post-processing for CREPE to turn f0 pitch estimates into discrete notes e.g. MIDI",
    entry_points={
        'console_scripts': [
            'crepe_notes=crepe_notes.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3.0 license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='crepe_notes',
    name='crepe_notes',
    packages=find_packages(include=['crepe_notes', 'crepe_notes.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/xavriley/crepe_notes',
    version='0.1.0',
    zip_safe=False,
)
