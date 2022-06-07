#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <iostream>

// Diese Funktion wollen wir in python aufrufen
void hello()
{
    std::cout << "hello from c" << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSIONS_NAME, io_module)
{
    // Functionspointer reinstekcen
    io_module.def("hello", &hello, "description"); 
}