import csv
import os, sys
from functools import reduce
import requests
import torch
import sympy as sp
import tempfile
import symbolicregression
from io import BytesIO

def load_model_from_url(device='cpu'):
    model_url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
    try:
        device = torch.device(device)

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
    unary_operators: list of unary operators (as strings) used in the predicted functions
    '''
    unary_set = {'abs', '**2', '**3', 'sqrt', 'sin', 'cos', 'tan', 'arctan','log', 'exp'}
    
    functions_str = ' '.join([obj.infix() for obj in predicted_functions]) # convert sympy functions to strings and join them
    unary_operators = set(string for string in unary_set if string in functions_str)

    
    unary_operators_dict = {'abs':'abs', '**2':'square', '**3':'cube', 'sqrt':'sqrt', 'sin':'sin', 'cos':'cos', 'tan':'tan', 'arctan':'atan','log':'log', 'exp':'exp'}
 
    updated_unary_operators = list(map(unary_operators_dict.get, unary_operators))

    return updated_unary_operators



def create_expressions(predicted_functions):
    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}

    def replace_operations(input_string, replace_ops):
        return reduce(lambda s, kv: s.replace(*kv), replace_ops.items(), input_string)

    # Extract the unary operators from the best function
    best_fx = predicted_functions[:2]
    best_uni_op = get_operators(best_fx)

    # Filter out equations with unary operators not in best_uni_op
    filtered_functions = (func for func in predicted_functions if all(op in best_uni_op for op in get_operators([func,])))

    # Apply the replacement function to each element in the 'filtered_functions' list
    expanded_functions = (sp.expand(replace_operations(func.infix(), replace_ops)) for func in filtered_functions)

    # Create a list of dictionaries with the specified columns using a list comprehension
    data = [{'equation': sp.sstr(equation), 'loss': 0, 'complexity': 1} for equation in expanded_functions]

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

def select_nested_constraints(nested_constraints, ops_list):
    selected_nested_constraints = {
        op: {
            sub_op: nested_constraints[op][sub_op] for sub_op in nested_constraints[op] if sub_op in ops_list
        }
        for op in ops_list if op in nested_constraints
    }
    return selected_nested_constraints


def select_constraints(constraints, ops_list):
    selected_constraints = {
        op: value for op, value in constraints.items() if op in ops_list
    }
    return selected_constraints