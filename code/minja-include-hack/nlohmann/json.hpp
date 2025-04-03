// This files serves as a solution to a problem where minja requires nlohmann::json which
// we already have in the alpaca core SDK, but the path is different and the minja's nlohmann include will not be found.
// That's why we add the include dir to this empty file which will be included by minja and compile successfully.
