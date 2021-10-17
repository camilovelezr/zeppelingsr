from setuptools import setup, find_packages

def fetch_requirements():
    required = []
    with open('requirements.txt') as f:
        required = f.read().splitlines()
    return required


PACKAGE_NAME = "zeppelingsr"

setup(
    name=PACKAGE_NAME,
    version='1.0.0',
    description='GSR Zeppelin Datasets for Cordra Upload',
    author='Camilo Velez',
    include_package_data=True,
    install_requires=fetch_requirements(),
    packages=find_packages()
)
