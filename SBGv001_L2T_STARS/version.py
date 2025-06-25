from os.path import join, abspath, dirname

# Read the version from a version.txt file located in the same directory as this script.
with open(join(abspath(dirname(__file__)), "version.txt")) as f:
    version = f.read().strip()  # .strip() to remove any leading/trailing whitespace

__version__ = version
__author__ = "Gregory H. Halverson, Evan Davis"
