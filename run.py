import argparse

from alg import *
from scut import *
from data import *

def main():
    args = parse_args()

    if args.d == 'sbm':
        graph_list = generate_graph(type='sbm')

    elif args.d == 'hsbm':
        graph_list = generate_graph(type='hsbm')
        
    elif args.d in ['IRIS','BOSTON','WINE','CANCER','NEWSGROUP']:
        G = generate_dataset_graph(args.d)
        graph_list = [G]
    else:
        raise Exception(f'Unknown experiment')

    epsilon_list = [0.01,0.1,0.5,1,2]
    our_alg_cost = run_alg("weight",graph_list,epsilon_list,args.s)
    ip_cost = run_alg("ip",graph_list,epsilon_list,args.s)
    nonpriv_cost = run_alg("non_priv",graph_list,epsilon_list,args.s)
    
    print("Dasgupta's cost by our algorithm: ", our_alg_cost)
    print("Dasgupta's cost by input perturbation: ", ip_cost)
    print("Dasgupta's cost non private algorithm: ", nonpriv_cost)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('-d', type=str, default='sbm', help="which data to test")
    parser.add_argument('-s', type=str, default='cheeger', help="which data to test")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()
