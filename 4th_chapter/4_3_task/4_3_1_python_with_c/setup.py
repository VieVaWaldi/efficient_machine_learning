import setuptools
import torch.utils.cpp_extension
import os

"""
    name = modulname
    ext_modules = unsere eigenen module
    -> Dann Modul und file name
    --> DANN in pip installieren:
    $ OPT="" pip3 install . -t $(pwd)/build
"""

setuptools.setup(name='eml_ext',
                 ext_modules=[
                     torch.utils.cpp_extension.CppExtension(
                         "eml_ext_hello_world_cpp",
                         ["hello.cpp"])],
                 cmdclass={
                     'build_ext': torch.utils.cpp_extension.BuildExtension}
                 )
