# Copyright 2018 - M. Pecheux
# [Forest Cover Type Classification] MAIN5 Machine Learning Project
# ------------------------------------------------------------------------------
# utils.py - Util functions
# ==============================================================================
import sys
    
def query_input(msg, default='no'):
    """Prompts a message with a yes/no answer and waits for the user to give an
    input. Defaults to "no"."""
    if default is None: prompt = ' [y/n] '
    elif default == 'yes': prompt = ' [Y/n] '
    elif default == 'no': prompt = ' [y/N] '
    else: raise ValueError('Invalid default answer: "{}".'.format(default))
    
    sys.stdout.write(msg + prompt)
    choice = input().lower()
    if choice == 'y' or choice == 'yes': return 1
    else: return 0
