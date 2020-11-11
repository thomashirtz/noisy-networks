from setuptools import setup

setup(
    name='noisynetworks',
    version='0.1.0',
    py_modules=['noisynetworks'],
    install_requires=['torch'],
    entry_points='''
        [console_scripts]
        noisynetworks=noisynetworks:noisynetworks
    '''
)
