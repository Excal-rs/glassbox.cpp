#include <cstdlib>
#include <iostream>
#include <string>
#include "glassbox/utils.h"

// --------- Public API ---------

// Prints "error: <msg>" to stderr and terminates the process with status 1.
void die(const std::string& msg)
{
    std::cerr << "error: " << msg << "\n";
    std::exit(1);
}
