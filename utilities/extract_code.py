"""
In this file, we define a function that extracts the code from a Python file,
To limit the code extracted in a comment make sure it has the string "non-essentials" in it.
To ignore specific functions or variables, add their names | pattern to the `to_be_ignored` set.
"""

import ast


def extract_code(start_documentation=True, file_name="", to_be_ignored=[]):
    response_lines = []

    # Read the current file and collect relevant lines
    with open(file_name, "r", encoding="utf-8") as f:
        documentation_start = False
        first = not start_documentation

        for line in f:
            stripped_line = line.strip()

            # Check if the line starts or ends the documentation string
            if first and '"""' in stripped_line:
                documentation_start = not documentation_start
                first = False
                continue

            if documentation_start and '"""' in stripped_line:
                documentation_start = not documentation_start
                continue

            if documentation_start and not start_documentation:
                continue

            # Stop collecting lines when reaching the specified comment
            if "#" in stripped_line and "non-essentials" in stripped_line:
                break

            # Collect the line
            response_lines.append(line)

    response = "".join(response_lines)

    # Parse the response with AST
    tree = ast.parse(response)

    # Filter out nodes based on the `to_be_ignored` set
    class CodeCleaner(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if any(pattern in node.name for pattern in to_be_ignored):
                return None
            return node

        def visit_Assign(self, node):
            if any(
                isinstance(target, ast.Name)
                and any(pattern in target.id for pattern in to_be_ignored)
                for target in node.targets
            ):
                return None
            return node

    # Transform the AST to remove ignored nodes
    cleaned_tree = CodeCleaner().visit(tree)

    # Convert the cleaned AST back to source code
    cleaned_code = ast.unparse(cleaned_tree)

    # Return the cleaned code
    return cleaned_code
