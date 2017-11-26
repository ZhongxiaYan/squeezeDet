import os

Src = os.path.dirname(os.path.abspath(__file__)) # src directory
Root = os.path.dirname(Src) + '/' # root directory
Src = Src + '/'
Models = os.path.join(Root, 'models') + '/'

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)