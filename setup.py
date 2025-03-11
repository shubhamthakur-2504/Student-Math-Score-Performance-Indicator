from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    '''This function will return the list of requirements'''
    requirement=[]
    with open(file_path) as file:
        requirement = file.readlines()
        requirement = [req.replace("\n","") for req in requirement]
        
        if HYPEN_E_DOT in requirement:
            requirement.remove(HYPEN_E_DOT)

    return requirement


setup(
    name='ml-project',
    version='0.0.1',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
   
    author='shubhamthakur-2504',
    author_email='shubhamthakur60278@gmail.com',
    python_requires='>=3.6',
)
