pybind11_add_module(_libdeath 
	bindings.cpp
	CountMatrix.cpp
)

install(TARGETS _libdeath 
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)