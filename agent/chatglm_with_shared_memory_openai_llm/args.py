import argparse
import os
from configs.model_config import *


# Additional argparse types
def path(string):
    if not string:
        return ''
    s = os.path.expanduser(string)
    if not os.path.exists(s):
        raise argparse.ArgumentTypeError(f'No such file or directory: "{string}"')
    return s


def file_path(string):
    if not string:
        return ''
    s = os.path.expanduser(string)
    if not os.path.isfile(s):
        raise argparse.ArgumentTypeError(f'No such file: "{string}"')
    return s


def dir_path(string):
    if not string:
        return ''
    s = os.path.expanduser(string)
    if not os.path.isdir(s):
        raise argparse.ArgumentTypeError(f'No such directory: "{string}"')
    return s


parser = argparse.ArgumentParser(prog='langchina-ChatGLM',
                                 description='基于langchain和chatGML的LLM文档阅读器')
parser.add_argument('-m', '--mode', default='demo', type=str, choices=['demo', 'web'],
                    help='Run demo in either single image demo mode (demo), web service mode (web), web client which '
                         'executes llm tasks '
                         'for a webserver (web_client) or batch translation mode (batch)')

parser.add_argument('--use-cuda', action='store_true', help='Turn on/off cuda')
parser.add_argument('--embedding-model', default=EMBEDDING_MODEL, type=str, choices=embedding_model_dict,
                    help='embedding_model')
parser.add_argument('--vector-search-top-k', default=6, type=int, help='vector_search_top_k')

parser.add_argument('--llm_model', default=LLM_MODEL, type=str, choices=llm_model_dict,
                    help='LLM model')
parser.add_argument('--llm-history-len', default=10, type=int, help='llm-history-len')
parser.add_argument('--dialogue-path', default='', type=str, help='dialogue-path')

# Generares dict with a default value for each argument
DEFAULT_ARGS = vars(parser.parse_args([]))
