from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph
import time
from sklearn.preprocessing import StandardScaler
# from sklearn.externals import joblib
import joblib
#version_info = list(map(int, nx.__version__.split('.')))
#major = version_info[0]
#minor = version_info[1]
#assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN=5
N_WALKS=50

def load_data(prefix, normalize=True, load_walks=False):
#def load_data(prefix, train_on_val_test, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    #if isinstance(G.nodes()[0], int):
        #conversion = lambda n : int(n)
        #print("True")
    #else:
        #conversion = lambda n : n
        #print("False")
    #conversion = lambda n : int(n)	
    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None		
    #id_map=json.load(open(prefix + "-id_map.json"))
    id_map_file = json.load(open(prefix + "-id_map.json"))# changed
    #print(id_map['0'])
    #id_map = {conversion(k):int(v) for k,v in id_map.items()}
    id_map = {v:int(v) for k,v in id_map_file.items()}
    #print(id_map['0'])
    #sys.exit(1)
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
	# Commented by Amine
    # if isinstance(list(class_map.values())[0], list):
        # lab_conversion = lambda n : n
    # else:
        # lab_conversion = lambda n : int(n)

    # class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}
    # class_map = {id_map_file[k]:lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    # broken_count = 0
    # for node in G.nodes():
        # if not 'val' in G.node[node] or not 'test' in G.node[node]:
            # G.remove_node(node)
            # broken_count += 1
    # print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():		
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        #from sklearn.preprocessing import StandardScaler
        #for n in G.nodes():
            #print(n)
            #print(id_map[n])
            #sys.exit(1)
        #if train_on_val_test==False:
        #train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_ids = np.array([id_map_file[str(n)] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        #else:
            #train_ids = np.array([id_map[n] for n in G.nodes()]) #tested by Amine
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    #return G, feats, id_map, walks, class_map
    #print(feats.shape)	
    return G, feats, id_map_file, walks, class_map

#added by Amine
def resource(filename):
    cwd = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cwd, filename)

#added by Amine
def load_data_chunks(prefix, log, data_prefix, training_thresh, max_nodes, normalize=True, load_walks=False):
    #chunks = [f for f in os.listdir(resource(prefix)) if os.path.isdir(os.path.join(prefix, f))]
    chunks = [os.path.join(prefix, f) for f in os.listdir(prefix) if os.path.isdir(os.path.join(prefix, f))]	
    chunks.sort()
    i=int(len(chunks) * training_thresh)
    j=int(len(chunks) * (training_thresh+((1-training_thresh)/2)))
    chunks_training = chunks[0 : int(len(chunks) * training_thresh)]
    chunks_validation = chunks[i:j]
    chunks_testing = chunks[j:]	
    graphs={}	
    graphs_train={}
    graphs_val={}
    graphs_test={}	
    feats_graphs=None
    #feats_graphs_test_val=None
    feats={}
    nbr_tests=0	
    id_maps={}
    class_maps={}
    #scaler = StandardScaler()
    flag=0	
    #walks ={}
    start_time = time.time()
    print("Loading Training Chunks ...")
    for item in chunks_training:
        G_data = json.load(open(item +os.sep+ data_prefix+"-G.json"))
        #G_data['test']=False
        #G_data['val']=False		
        G = json_graph.node_link_graph(G_data)
        graphs_train[G.name]=G
        graphs[G.name]=G		
		
        if os.path.exists(item +os.sep+ data_prefix+"-feats.npy"):
            #feats = np.load(item + os.sep+data_prefix+"-feats.npy")
            feats[G.name] = np.load(item + os.sep + data_prefix+"-feats.npy")			
            #print(feats[G.name].shape)			
            pad= -1*np.ones((max_nodes-feats[G.name].shape[0],feats[G.name].shape[1])) #missing values with respect to max nodes
            #print(pad.shape)
            #sys.exit(1)			
            feats[G.name]= np.vstack((feats[G.name],pad))		
            if feats_graphs is None:
                feats_graphs=feats[G.name]
            else:
                feats_graphs = np.vstack((feats_graphs, feats[G.name]))			
            #feats_graphs[item]=feats
        else:
            print("No features present.. Only identity features will be used.")
            #feats = None
            #feats_graphs[item]=feats
        id_map_file = json.load(open(item + os.sep+data_prefix+"-id_map.json"))# changed

        #id_map = {v:int(v) for k,v in id_map_file.items()}
        #id_maps[item] = id_map_file
        id_maps[G.name] = id_map_file		
        #walks[item]= []
        class_map = json.load(open(item + os.sep+data_prefix+"-class_map.json"))
        class_maps[G.name]=class_map
        if flag==0:
            num_classes=len(list(class_map.values())[0])
            flag = 1			
        #print("Loaded data.. now preprocessing..")
        # for edge in G.edges():		
            # if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                # G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
                # G[edge[0]][edge[1]]['train_removed'] = True
            # else:
                # G[edge[0]][edge[1]]['train_removed'] = False
    print("--- Training chunks %s seconds ---" % (time.time() - start_time))    
    print("Loading Testing Chunks ...")
    start_time = time.time()
    for item in chunks_testing:
        G_data = json.load(open(item +os.sep+ data_prefix+ "-G.json"))
        #G_data['test']=True
        #G_data['val']=False
        G = json_graph.node_link_graph(G_data)
        graphs_test[G.name]=G
        graphs[G.name]=G		
        if os.path.exists(item + os.sep+data_prefix+"-feats.npy"):
            feats[G.name] = np.load(item + os.sep+data_prefix+"-feats.npy")
            pad= -1*np.ones((max_nodes-feats[G.name].shape[0],feats[G.name].shape[1])) #missing values with respect to max nodes
            feats[G.name]= np.vstack((feats[G.name],pad))			
            if feats_graphs is None:
                feats_graphs = feats[G.name]
            else:
                feats_graphs = np.vstack((feats_graphs, feats[G.name]))
            
        #G_data['test']=False
        #G_data['val']=False		
        #G = json_graph.node_link_graph(G_data)
        # graphs[item]=G
        # if os.path.exists(item + "-feats.npy"):
            # feats = np.load(item + "-feats.npy")
            # if feats_graphs_test_val == None:
                # feats_graphs_test_val=feats
            # else:
                # feats_graphs_test_val = np.vstack((feats_graphs_test_val, feats))	
            #feats_graphs_test_val[item]=feats
        #else:
            #print("No features present.. Only identity features will be used.")
            #feats = None
            #feats_graphs_test_val[item]=feats

        id_map_file = json.load(open(item + os.sep+data_prefix+"-id_map.json"))# changed

        #id_map = {v:int(v) for k,v in id_map_file.items()}
        id_maps[G.name] = id_map_file
        #walks[item]= []
        class_map = json.load(open(item + os.sep+data_prefix+"-class_map.json"))
        class_maps[G.name]=class_map 		
        #print("Loaded data.. now preprocessing..")
        # for edge in G.edges():		
            # if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                # G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
                # G[edge[0]][edge[1]]['train_removed'] = True
            # else:
                # G[edge[0]][edge[1]]['train_removed'] = False
    print("--- Testing chunks %s seconds ---" % (time.time() - start_time))
    print("Loading Validation Chunks ...")	
    start_time = time.time()
    for item in chunks_validation:
        G_data = json.load(open(item + os.sep+data_prefix+"-G.json"))
        #G_data['test']=False
        #G_data['val']=True
        G = json_graph.node_link_graph(G_data)
        graphs_val[G.name]=G
        graphs[G.name]=G			
        if os.path.exists(item + os.sep+ data_prefix+"-feats.npy"):
            feats[G.name] = np.load(item + os.sep+data_prefix+"-feats.npy")
            #feats = np.load(item + os.sep+ data_prefix+ "-feats.npy")
            pad= -1*np.ones((max_nodes-feats[G.name].shape[0],feats[G.name].shape[1])) #missing values with respect to max nodes
            feats[G.name]= np.vstack((feats[G.name],pad))			
            if feats_graphs is None:
                feats_graphs = feats[G.name]
            else:
                feats_graphs = np.vstack((feats_graphs, feats[G.name]))
            				
        #else:
            #print("No features present.. Only identity features will be used.")
            #feats = None
            #feats_graphs_test_val[item]=feats

        id_map_file = json.load(open(item +os.sep+ data_prefix+"-id_map.json"))# changed

        #id_map = {v:int(v) for k,v in id_map_file.items()}
        id_maps[G.name] = id_map_file
        #walks[item]= []
        class_map = json.load(open(item + os.sep+ data_prefix+"-class_map.json"))
        class_maps[G.name]=class_map 		
        #print("Loaded data.. now preprocessing..")
        # for edge in G.edges():		
            # if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                # G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
                # G[edge[0]][edge[1]]['train_removed'] = True
            # else:
                # G[edge[0]][edge[1]]['train_removed'] = False
    print("--- Validation chunks %s seconds ---" % (time.time() - start_time))
    if normalize and not feats is None:
        print("--- Start Normalization ---")        
        #train_ids = np.array([id_map_file[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        #else:
            #train_ids = np.array([id_map[n] for n in G.nodes()])		
        #train_feats = feats[train_ids]		
        scaler = StandardScaler()
        scaler.fit(feats_graphs)	
        for item in feats.keys():
            feats[item]=scaler.transform(feats[item])		
        joblib.dump(scaler, log) 		
        #scaler.fit(feats_graphs)
        #feats_graphs = scaler.transform(feats_graphs)
        #feats = scaler.transform(np.vstack((feats_graphs,feats_graphs_test_val)))    
        #if load_walks:
            #with open(prefix + "-walks.txt") as fp:
                #for line in fp:
                    #walks.append(map(conversion, line.split()))
    #for item in feats.keys():
        #print(feats[item].shape)
        #print(feats[item])
    #sys.exit(1)	
    return graphs, graphs_train, graphs_val, graphs_test, feats,id_maps,class_maps,num_classes#,scaler

#added by Amine
def load_graph_test(scaler, prefix, data_prefix, max_nodes, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + os.sep+data_prefix+"-G.json"))
    G = json_graph.node_link_graph(G_data)
    feats = None
    if os.path.exists(prefix + os.sep + data_prefix+"-feats.npy"):
        feats = np.load(prefix + os.sep + data_prefix+"-feats.npy")					
        pad= -1*np.ones((max_nodes-feats.shape[0],feats.shape[1])) #missing values with respect to max nodes
        feats = np.vstack((feats,pad))	
    id_map_file = json.load(open(prefix + os.sep+data_prefix+"-id_map.json"))# changed
    class_map = json.load(open(prefix + os.sep+ data_prefix+"-class_map.json"))
    if normalize and not feats is None:
        print("--- Start Normalization ---")
        feats=scaler.transform(feats)
    return G, feats, id_map_file, class_map		
		
def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
