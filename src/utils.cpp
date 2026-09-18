// Include `glassbox` libraries
#include "glassbox/utils.h"

// Include stdlib
#include <cstdlib>
#include <iostream>
#include <string>

// --------- Public API ---------

// Prints "error: <msg>" to stderr and terminates the process with status 1.
void die(const std::string& msg)
{
    std::cerr << "error: " << msg << "\n";
    std::exit(1);
}
