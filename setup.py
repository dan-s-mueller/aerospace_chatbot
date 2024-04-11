import setuptools
import toml

# Load the project metadata from pyproject.toml
pyproject_path = 'pyproject.toml'
with open(pyproject_path, 'r', encoding='utf-8') as f:
    pyproject = toml.load(f)

# Extract project metadata
metadata = pyproject['tool']['poetry']

# Use the metadata in setuptools.setup
setuptools.setup(
    name=metadata['name'],
    version=metadata['version'],
    author=metadata['authors'],
    description=metadata['description'],
    packages=setuptools.find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires=metadata['dependencies']['python']
    install_requires=metadata['dependencies']
)
