import os
from setuptools import setup, find_packages, __version__


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


with open('requirements.txt') as f:
    required_packages = f.readlines()

setup(
    name='comet',
    version=__version__,
    description='Meteor for training, evaluating and interring SOTA models',
    packages=find_packages(),
    include_package_data=True,
    py_modules=['comet'],
    install_requires=required_packages,
    python_requires='>3.7.0',
    author='phongnt',
    author_email='phongnt@ftech.ai',

    entry_points={
        'console_scripts': [
            'meteor = meteor.run_cli:entry_point'
        ]
    },
)
