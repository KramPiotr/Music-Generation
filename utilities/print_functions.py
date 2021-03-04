# -*- coding: future_fstrings -*-
import ast
import sys
import os

directory = "."
file = "utility_description.txt"

def top_level_functions(body):
    return (f for f in body if isinstance(f, ast.FunctionDef))

def parse_ast(filename):
    with open(filename, "rt") as file:
        return ast.parse(file.read(), filename=filename)

if __name__ == "__main__":
    with open(file, "w") as f:
        for filename in os.listdir(directory):
            if filename.startswith("__") or not filename.endswith(".py"):
                continue
            print(filename, file=f)
            print(filename)
            filename = os.path.join(directory, filename)
            tree = parse_ast(filename)
            for func in top_level_functions(tree.body):
                print(f"  {func.name}", file=f)
            print("",file=f)

