import sys
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

print(sys.version)
from setuptools import setup, find_packages

class CustomBuildCommand(build_py):
    def run(self):
        # 在构建之前清理 build 目录
        self.clean_build_directory()
        super().run()

    def clean_build_directory(self):
        import shutil
        shutil.rmtree('build')

setup(
    name='highway-env',
    version='1.0.dev0',
    description='An environment for simulated highway driving tasks',
    url='https://github.com/eleurent/highway-env',
    author='Edouard Leurent',
    author_email='eleurent@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='autonomous highway driving simulation environment reinforcement learning',
    packages=find_packages(exclude=['docs', 'scripts', 'tests']),
    install_requires=['gym', 'numpy', 'pygame', 'matplotlib', 'pandas'],
    tests_require=['pytest'],
    extras_require={
        'dev': ['scipy'],
        'deploy': ['pytest-runner', 'sphinx<1.7.3', 'sphinx_rtd_theme']
    },
    entry_points={
        'console_scripts': [],
    },
)

