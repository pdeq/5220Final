# mylib.py
import ctypes

# Load the shared library (compiled from mylib.c)
mylib = ctypes.CDLL('./mylib.so')

# Define the Python wrapper function
def my_fputs_wrapper(s):
    mylib.my_fputs(s.encode())

# Example usage
my_fputs_wrapper("Hello from Python!\n")
