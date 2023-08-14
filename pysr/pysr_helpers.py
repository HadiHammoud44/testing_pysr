import csv
import os, sys
from functools import reduce
import requests
import torch
import sympy
import tempfile
import symbolicregression
from io import BytesIO

def load_model_from_url():
    model_url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        response = requests.get(model_url)
        model_binary = response.content

        model = torch.load(BytesIO(model_binary), map_location=device)
        model.to(device)
        return model

    except Exception as e:
        print("ERROR: Model not loaded! Error details:")
        print(e) 
        return None


def get_operators(predicted_functions):
    '''
    input: 
    predicted_functions: nodelist

    output:
    binary_operators: list of binary operators (as strings) used in the predicted functions
    unary_operators: list of unary operators (as strings) used in the predicted functions
    '''
    unary_set = {'add', 'sub', 'mul', 'inv'}
    binary_set = {'abs', '**2', '**3', 'sqrt', 'sin', 'cos', 'tan', 'arctan','log', 'exp'}
    
    functions_str = ' '.join([obj.infix() for obj in predicted_functions]) # convert sympy functions to strings and join them
    binary_operators = set(string for string in unary_set if string in functions_str)
    unary_operators = set(string for string in binary_set if string in functions_str)

    
    unary_operators_dict = {'abs':'abs', '**2':'square', '**3':'cube', 'sqrt':'sqrt', 'sin':'sin', 'cos':'cos', 'tan':'tan', 'arctan':'atan','log':'log', 'exp':'exp'}
    binary_operators_dict = {'add':'+', 'sub':'-', 'mul':'*', 'inv':'/'}
 
    updated_binary_operators = list(map(binary_operators_dict.get, binary_operators))
    updated_unary_operators = list(map(unary_operators_dict.get, unary_operators))

    return updated_unary_operators, updated_binary_operators



def create_expressions(predicted_functions):
    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
    
    def replace_operations(input_string, replace_ops):
        return reduce(lambda s, kv: s.replace(*kv), replace_ops.items(), input_string)

    # Apply the replacement function to each element in the 'result_list'
    replaced_functions = [replace_operations(function_expression.infix(), replace_ops) for function_expression in predicted_functions]

    simplified_functions = (sympy.expand(fx) for fx in replaced_functions)

    # Create a list of dictionaries with the specified columns using a list comprehension
    data = [{'equation': sympy.sstr(equation), 'loss': 1, 'complexity': 1} for equation in simplified_functions]

    # Create a temporary in-memory CSV file
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False, newline='') as temp_csv:
        fieldnames = ['equation', 'loss', 'complexity']
        writer = csv.DictWriter(temp_csv, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write the data rows
        writer.writerows(data)

        # Move the file cursor to the beginning for reading
        temp_csv.seek(0)

    return temp_csv