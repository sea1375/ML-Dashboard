from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import os

np.random.seed(123)

class EdgeMinibatchIterator(object):
    
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    G -- networkx graph
    id2idx -- dict mapping node ids to index in feature tensor
    placeholders -- tensorflow placeholders object
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    n2v_retrain -- signals that the iterator is being used to add new embeddings to a n2v model
    fixed_n2v -- signals that the iterator is being used to retrain n2v with only existing nodes as context
    """
    def __init__(self, G, id2idx, 
            placeholders, context_pairs=None, batch_size=100, max_degree=25,
            n2v_retrain=False, fixed_n2v=False,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0

        self.nodes = np.random.permutation(G.nodes())
        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()
        if context_pairs is None:
            edges = G.edges()
        else:
            edges = context_pairs
        self.train_edges = self.edges = np.random.permutation(edges)
        if not n2v_retrain:
            self.train_edges = self._remove_isolated(self.train_edges)
            self.val_edges = [e for e in G.edges() if G[e[0]][e[1]]['train_removed']]
        else:
            if fixed_n2v:
                self.train_edges = self.val_edges = self._n2v_prune(self.edges)
            else:
                self.train_edges = self.val_edges = self.edges

        print(len([n for n in G.nodes() if not G.node[n]['test'] and not G.node[n]['val']]), 'train nodes')
        print(len([n for n in G.nodes() if G.node[n]['test'] or G.node[n]['val']]), 'test nodes')
        self.val_set_size = len(self.val_edges)

    def _n2v_prune(self, edges):
        is_val = lambda n : self.G.node[n]["val"] or self.G.node[n]["test"]
        return [e for e in edges if not is_val(e[1])]

    def _remove_isolated(self, edge_list):
        new_edge_list = []
        missing = 0
        for n1, n2 in edge_list:
            if not n1 in self.G.node or not n2 in self.G.node:
                missing += 1
                continue
            if (self.deg[self.id2idx[n1]] == 0 or self.deg[self.id2idx[n2]] == 0) \
                    and (not self.G.node[n1]['test'] or self.G.node[n1]['val']) \
                    and (not self.G.node[n2]['test'] or self.G.node[n2]['val']):
                continue
            else:
                new_edge_list.append((n1,n2))
        print("Unexpected missing:", missing)
        return new_edge_list

    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_edges)

    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})

        return feed_dict

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges))
        batch_edges = self.train_edges[start_idx : end_idx]
        return self.batch_feed_dict(batch_edges)

    def num_training_batches(self):
        return len(self.train_edges) // self.batch_size + 1

    def val_feed_dict(self, size=None):
        edge_list = self.val_edges
        if size is None:
            return self.batch_feed_dict(edge_list)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges)

    def incremental_val_feed_dict(self, size, iter_num):
        edge_list = self.val_edges
        val_edges = edge_list[iter_num*size:min((iter_num+1)*size, 
            len(edge_list))]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(self.val_edges), val_edges

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size, 
            len(node_list))]
        val_edges = [(n,n) for n in val_nodes]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(node_list), val_edges

    def label_val(self):
        train_edges = []
        val_edges = []
        for n1, n2 in self.G.edges():
            if (self.G.node[n1]['val'] or self.G.node[n1]['test'] 
                    or self.G.node[n2]['val'] or self.G.node[n2]['test']):
                val_edges.append((n1,n2))
            else:
                train_edges.append((n1,n2))
        return train_edges, val_edges

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_edges = np.random.permutation(self.train_edges)
        self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0

class NodeMinibatchIterator(object):
    
    """ 
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, G, id2idx, 
            placeholders, label_map, num_classes, 
            batch_size=100, max_degree=25,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.label_map = label_map
        self.num_classes = num_classes

        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()

        self.val_nodes = [n for n in self.G.nodes() if self.G.node[n]['val']]
        self.test_nodes = [n for n in self.G.nodes() if self.G.node[n]['test']]

        self.no_train_nodes_set = set(self.val_nodes + self.test_nodes)
        self.train_nodes = set(G.nodes()).difference(self.no_train_nodes_set)
		#changed by Amine
        #self.train_nodes = self.nodes()
        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.deg[id2idx[str(n)]] > 0]

    def _make_label_vec(self, node):
        label = self.label_map[str(node)]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[str(node)]
            label_vec[class_ind] = 1
        return label_vec

    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbors = np.array([self.id2idx[str(neighbor)] 
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])])
            deg[self.id2idx[str(nodeid)]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            #print(len(self.id2idx))				
            #print(neighbors)
            #print(deg.shape)
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            #print(neighbors)
            #print(neighbors.shape)			
            adj[self.id2idx[str(nodeid)], :] = neighbors
            #print(adj[self.id2idx[str(nodeid)], :])
            #sys.exit(1)
        #print(adj.shape)
        #print(deg.shape)
        #sys.exit(1)		
        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[str(neighbor)] 
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[str(nodeid)], :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes, val=False):
        batch1id = batch_nodes
        batch1 = [self.id2idx[str(n)] for n in batch1id]	
        labels = np.vstack([self._make_label_vec(node) for node in batch1id])
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch1)})
        feed_dict.update({self.placeholders['batch']: batch1})
        feed_dict.update({self.placeholders['labels']: labels})

        return feed_dict, labels

    def node_val_feed_dict(self, size=None, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
	
        if not size is None:
            val_nodes = np.random.choice(val_nodes, size, replace=True)
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_nodes)
        return ret_val[0], ret_val[1]

    def incremental_node_val_feed_dict(self, size, iter_num, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        val_node_subset = val_nodes[iter_num*size:min((iter_num+1)*size, 
            len(val_nodes))]

        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_node_subset)
        return ret_val[0], ret_val[1], (iter_num+1)*size >= len(val_nodes), val_node_subset

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx : end_idx]
        return self.batch_feed_dict(batch_nodes)

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size, 
            len(node_list))]
        return self.batch_feed_dict(val_nodes), (iter_num+1)*size >= len(node_list), val_nodes

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

# added by Amine

class GraphMinibatchIterator(object): 

    def __init__(self, Graphs, Graphs_train, Graphs_val, Graphs_test, id2idxs, placeholders,
            label_maps, num_classes, 
            batch_size=100, max_degree=25,
            **kwargs):
		
        self.Graphs = Graphs
        self.Graphs_train = Graphs_train
        self.Graphs_val = Graphs_val	
        self.Graphs_test = Graphs_test		
        #self.nodes = G.nodes()
        self.id2idxs = id2idxs
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.label_maps = label_maps
        self.num_classes = num_classes
        self.train_instances=0		

        #self.adj, self.deg = self.construct_adj()
        #self.test_adj = self.construct_test_adj()
        self.adj, self.deg = self.construct_adj_ids()
        self.test_adj = self.construct_test_adj_ids()		
        #self.test_adj = self.adj	
        self.val_nodes = {}
        self.test_nodes = {}
        self.train_nodes = {}
        self.train_val_graphs=[]
        self.test_graphs=[]
        #self.train_nodes_length = 0
		
        # for G in self.Graphs:
            # if G['val']:
                # l = [n for n in G.nodes() if G['val']]
                # if self.val_nodes!={}:
                    # self.val_nodes[G["id"]].extend(l)
                # else:
                    # self.val_nodes[G["id"]] = l
                # self.train_val_graphs.append(G)					
            # elif G['test']:
                # l = [n for n in G.nodes() if G['test']]
                # if self.test_nodes!={}:
                    # self.test_nodes[G["id"]].extend(l)
                # else:
                    # self.test_nodes[G["id"]] = l
                # self.test_graphs.append(G)						
            # else:
                # l = [n for n in G.nodes() if not G['test'] and not G['val']]
                # if self.train_nodes!={}:
                    # self.train_nodes_length=self.train_nodes_length+len(l)
                    # #self.train_nodes[G["id"]].extend(l)
                # else:
                    # self.train_nodes[G["id"]] = l
                # self.train_val_graphs.append(G)					
                    # #self.train_nodes_length = len(l)
	
        for k in Graphs_train.keys():
            #print(k)
            #sys.exit(1)			
            self.train_nodes[Graphs_train[k].name] = [n for n in Graphs_train[k].nodes()]
            self.train_val_graphs.append(Graphs_train[k])
            self.train_instances=self.train_instances+1			
        for k in Graphs_test.keys():
            self.test_nodes[Graphs_test[k].name] = [n for n in Graphs_test[k].nodes()]
            self.test_graphs.append(Graphs_test[k])			
        for k in Graphs_val.keys():
            self.val_nodes[Graphs_val[k].name] = [n for n in Graphs_val[k].nodes()]
            self.train_val_graphs.append(Graphs_val[k])			

    def _make_label_vec(self, id, node):
        #print(self.label_maps)
        #print(id)
        #print(node)
        #sys.exit(1)		
        label = self.label_maps[id][str(node)]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_maps[id][str(node)]
            label_vec[class_ind] = 1
        #print(label_vec)
        #input()		
        return label_vec

		
    def construct_adj(self):
        adj = None
        deg = None		
        flag = 0		
        for item in self.id2idxs.keys():
            tmp = len(self.id2idxs[item])*np.ones((len(self.id2idxs[item])+1, self.max_degree))		
            tmp_deg = np.zeros((len(self.id2idxs[item]),)) 
            #k=item[item.rfind[os.sep]+1:] #get key to get the right graph			
            for nodeid in self.Graphs[item].nodes:
                #print(item)
                #print(nodeid)
                #print(self.Graphs[item].neighbors(nodeid))				
                #for neighbor in self.Graphs[item].neighbors(nodeid):
                    #print(neighbor)
                    #print(self.id2idxs[item])					
                #sys.exit(1)					
                neighbors = np.array([self.id2idxs[item][str(neighbor)] 
                    for neighbor in self.Graphs[item].neighbors(nodeid)])
                #print(neighbors)
                #sys.exit(1)
                tmp_deg[self.id2idxs[item][str(nodeid)]] = len(neighbors)
			
                if len(neighbors) == 0:
                    continue
                if len(neighbors) > self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
                elif len(neighbors) < self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
                tmp[self.id2idxs[item][str(nodeid)], :] = neighbors
            if flag==0:
                adj = tmp
                deg = tmp_deg
                flag = 1
            else:
                sh1 = tmp.shape
                sh0	= adj.shape		
                adj = np.hstack((adj,np.zeros((sh0[0],sh1[1]), dtype=int)))
                tmp = np.hstack((np.zeros((sh1[0],sh0[1]), dtype=int),tmp))
                adj = np.vstack((adj,tmp))				
                deg = np.hstack((deg,tmp_deg))				
        return adj, deg
		
    def construct_test_adj(self):
        adj = None	
        flag = 0		
        for item in self.id2idxs.keys():
            tmp = len(self.id2idxs[item])*np.ones((len(self.id2idxs[item])+1, self.max_degree)) 
            #k=item[item.rfind[os.sep]+1:] #get key to get the right graph			
            for nodeid in self.Graphs[item].nodes:
                neighbors = np.array([self.id2idxs[item][str(neighbor)] 
                    for neighbor in self.Graphs[item].neighbors(nodeid)])
                if len(neighbors) == 0:
                    continue
                if len(neighbors) > self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
                elif len(neighbors) < self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
                tmp[self.id2idxs[item][str(nodeid)], :] = neighbors
            if flag==0:
                adj = tmp
                flag = 1
            else:
                sh1 = tmp.shape
                sh0	= adj.shape		
                adj = np.hstack((adj,np.zeros((sh0[0],sh1[1]), dtype=int)))
                tmp = np.hstack((np.zeros((sh1[0],sh0[1]), dtype=int),tmp))
                adj = np.vstack((adj,tmp))			
        return adj

    def construct_adj_ids(self):
        adj = None
        deg = None	
        #flag = 0
        adjs={}
        degs={}	
        for item in self.id2idxs.keys():
            adj = len(self.id2idxs[item])*np.ones((len(self.id2idxs[item])+1, self.max_degree))		
            deg = np.zeros((len(self.id2idxs[item]),)) 
            #k=item[item.rfind[os.sep]+1:] #get key to get the right graph
            #if item in self.Graphs_train.keys():
            if item in self.Graphs_train.keys() or item in self.Graphs_val.keys():			
                #for nodeid in self.Graphs_train[item].nodes:
                for nodeid in self.Graphs[item].nodes:				
                    #print(item)
                    #print(nodeid)
                    #print(self.Graphs[item].neighbors(nodeid))				
                    #for neighbor in self.Graphs[item].neighbors(nodeid):
                        #print(neighbor)
                        #print(self.id2idxs[item])					
                    #sys.exit(1)					
                    #neighbors = np.array([self.id2idxs[item][str(neighbor)] 
                        #for neighbor in self.Graphs_train[item].neighbors(nodeid)])
                    neighbors = np.array([self.id2idxs[item][str(neighbor)] 
                        for neighbor in self.Graphs[item].neighbors(nodeid)])						
                    #print(neighbors)
                    #sys.exit(1)
                    deg[self.id2idxs[item][str(nodeid)]] = len(neighbors)
			
                    if len(neighbors) == 0:
                        continue
                    if len(neighbors) > self.max_degree:
                        neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
                    elif len(neighbors) < self.max_degree:
                        neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
                    adj[self.id2idxs[item][str(nodeid)], :] = neighbors
                #print(adj.shape)
                #print(deg.shape)				
                #sys.exit(1)			
                adjs[item]=adj
                degs[item]=deg
        #sys.exit(1)				
        return adjs, degs

    def construct_test_adj_ids(self):
        adjs={}	
        for item in self.id2idxs.keys():
            adj = len(self.id2idxs[item])*np.ones((len(self.id2idxs[item])+1, self.max_degree))
            if item in self.Graphs_test.keys():			
                #k=item[item.rfind[os.sep]+1:] #get key to get the right graph			
                for nodeid in self.Graphs_test[item].nodes:
                    neighbors = np.array([self.id2idxs[item][str(neighbor)] 
                        for neighbor in self.Graphs_test[item].neighbors(nodeid)])
                    if len(neighbors) == 0:
                        continue
                    if len(neighbors) > self.max_degree:
                        neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
                    elif len(neighbors) < self.max_degree:
                        neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
                    adj[self.id2idxs[item][str(nodeid)], :] = neighbors
                adjs[item]=adj
        return adjs

    
    def end(self,id): #id is graph id
        #print(self.batch_num * self.batch_size )
        #print(len(self.train_nodes[id]))	
        return self.batch_num * self.batch_size >= len(self.train_nodes[id])

    def batch_feed_dict(self, batch_nodes, id, val=False):
        #batch1id = batch_nodes
        #print(id)
        #print(len(batch_nodes))
        #for n in batch_nodes:
            #print(self.id2idxs[id][str(n)])		
        #sys.exit(1)		
        batch1 = [self.id2idxs[id][str(n)] for n in batch_nodes]		
        labels = np.vstack([self._make_label_vec(id,node) for node in batch_nodes])
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch1)})
        feed_dict.update({self.placeholders['batch']: batch1})
        feed_dict.update({self.placeholders['labels']: labels})

        return feed_dict, labels
				

    def node_val_feed_dict(self, id, size=None): #id is graph id
        if test:
            val_nodes = self.test_nodes[id]
        else:
            val_nodes = self.val_nodes[id]
	
        if not size is None:
            val_nodes = np.random.choice(val_nodes, size, replace=True)
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_nodes,id)
        return ret_val[0], ret_val[1]

    def incremental_node_val_feed_dict(self, id, size, iter_num, test=False): #id is graph id
        if test:
            val_nodes = self.test_nodes[id]
        else:
            val_nodes = self.val_nodes[id]
        #print(type(val_nodes))				
        val_node_subset = val_nodes[iter_num*size:min((iter_num+1)*size, len(val_nodes))]

        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_node_subset,id)
        return ret_val[0], ret_val[1], (iter_num+1)*size >= len(val_nodes), val_node_subset

    def num_training_batches(self, id): #id is graph id
        return len(self.train_nodes[id]) // self.batch_size + 1

    def next_minibatch_feed_dict(self,id):#id is graph id
        #print(self.batch_num)	
        #print(self.batch_size)	
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        #print(self.train_nodes[id])		
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes[id]))
        batch_nodes = self.train_nodes[id][start_idx : end_idx]
        #print(batch_nodes)		
        #print(self.batch_feed_dict(batch_nodes,id))
        #input()		
        return self.batch_feed_dict(batch_nodes,id)

    def incremental_embed_feed_dict(self, id, size, iter_num): #id is graph id
        node_list = self.nodes[id]
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size, 
            len(node_list))]
        return self.batch_feed_dict(val_nodes,id), (iter_num+1)*size >= len(node_list), val_nodes

    def shuffle(self,id): #id is graph id
        """ Re-shuffle the training set.
            Also reset the batch number.
        """		
        self.train_nodes[id] = np.random.permutation(self.train_nodes[id])
        self.batch_num = 0
		
