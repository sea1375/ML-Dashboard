from __future__ import division
from __future__ import print_function

import os,sys
import time
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics
# from sklearn.externals import joblib
import joblib
import pickle

from graphsage.supervised_models import SupervisedGraphsage,SupervisedGraphsagePlus
from graphsage.models import SAGEInfo
from graphsage.minibatch import NodeMinibatchIterator
from graphsage.minibatch import GraphMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import load_data
from graphsage.utils import load_data_chunks
from graphsage.utils import load_graph_test

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,"""Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'graphsage_mean', 'model names. See README for possible values.')  
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')
#flags.DEFINE_boolean('all_neighbors', False, 'train on chuncked data located under train_prefix') #added by Amine Boukhtouta, select all neighbors 
flags.DEFINE_boolean('train_chunks', False, 'train on chuncked data located under train_prefix') #added by Amine Boukhtouta
flags.DEFINE_boolean('train_mode', True, 'train on chuncked data located under train_prefix') #added by Amine Boukhtouta
flags.DEFINE_string('data_prefix', 'adpcicd', 'data prefix') #added by Amine Boukhtouta
#flags.DEFINE_boolean('train_on_val_test', False, 'training on data considered for validation and testing') #added by Amine Boukhtouta
flags.DEFINE_float('train_percentage', 0.8, 'Percentage of training graphs') #added by Amine
flags.DEFINE_integer('nodes_max', 400, 'Maximum Number of Nodes') #added by Amine
# left to default values in main experiments 
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
#flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
#flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
#flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('samples_1', 8, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 4, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_string('model_dir', '.', 'directory of a saved model') #added by Amine
flags.DEFINE_string('test_dir','.', 'test directory after saving a model') #added by Amine
#flags.DEFINE_boolean('pad_features', True, 'pad features when testing') #added by Amine
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    #metrics.f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")
    #return metrics.f1_score(y_true, y_pred, average="micro", labels=np.unique(y_pred)), metrics.f1_score(y_true, y_pred, average="macro", labels=np.unique(y_pred))

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    #print(minibatch_iter.val_nodes)
    #sys.exit(1)
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss], 
                        feed_dict=feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0])
    return node_outs_val[1], mic, mac, (time.time() - t_test)
	
	
# added by Amine
def evaluate_graph(sess, model, minibatch_iter, graph_id, size=None):
    t_test = time.time()
    #print(minibatch_iter.val_nodes)
    #sys.exit(1)
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(graph_id,size)
    node_outs_val = sess.run([model.preds, model.loss], 
                        feed_dict=feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0])
    return node_outs_val[1], mic, mac, (time.time() - t_test)

def log_dir():
    log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def incremental_evaluate(sess, model, minibatch_iter, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    while not finished:
        feed_dict_val, batch_labels, finished, _  = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
        node_outs_val = sess.run([model.preds, model.loss], 
                         feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)


#added by Amine
def incremental_evaluate_graph(sess, model, minibatch_iter, graph_id, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _  = minibatch_iter.incremental_node_val_feed_dict(graph_id, size, iter_num, test=test)
        node_outs_val = sess.run([model.preds, model.loss], 
                         feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)
	

def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels' : tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders


def train(train_data, test_data=None):

    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map  = train_data[4]
    #print(G)
    #print(features.shape)
    #print(len(id_map.keys()))
    #print(class_map)
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))
    #print(num_classes)
    #sys.exit(1)
    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])
        #print(features.shape) 		
        #sys.exit(1)

    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders(num_classes)
    #print(context_pairs)
    #sys.exit(1)	
    minibatch = NodeMinibatchIterator(G, 
            id_map,
            placeholders, 
            class_map,
            num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree, 
            context_pairs = context_pairs)
    #print(minibatch.adj.shape)
    #input()	
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    #print(adj_info_ph)
    #sys.exit(1)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
    #print(adj_info)
    #sys.exit(1)
    #print(features.shape)
    #input()	
    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        #print(sampler)
        #sys.exit(1)		
        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos, 
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    elif FLAGS.model == 'gcn':
        # Create model
        if FLAGS.all_neighbors == False:		
            sampler = UniformNeighborSampler(adj_info)
            #print(sampler.adj_info)
            #sys.exit(1)			
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]
            model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="gcn",
                                     model_size=FLAGS.model_size,
                                     concat=False,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]

            model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="gcn",
                                     model_size=FLAGS.model_size,
                                     concat=False,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)		

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="seq",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="maxpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="meanpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True
    
    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
     
    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
    
    # Train model
    
    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    #print(train_adj_info)	
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    #print(val_adj_info)
    #sys.exit(1)
    for epoch in range(FLAGS.epochs): 
        minibatch.shuffle() 

        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict, labels = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            #print(feed_dict)
            #sys.exit(1)			
            t = time.time()
            # Training step
            #print(merged)
            #print(model.opt_op)
            #print(model.loss)
            #print(model.preds)
            #print(feed_dict)
            #print(labels)					
            #input()				
            outs = sess.run([merged, model.opt_op, model.loss, model.preds], feed_dict=feed_dict)
            train_cost = outs[2]

            if iter % FLAGS.validate_iter == 0:
                # Validation
                sess.run(val_adj_info.op)
                if FLAGS.validate_batch_size == -1:
                    val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
                else:
                    val_cost, val_f1_mic, val_f1_mac, duration = evaluate(sess, model, minibatch, FLAGS.validate_batch_size)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)
    
            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1])
                print("Iter:", '%04d' % iter, 
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_f1_mic=", "{:.5f}".format(train_f1_mic), 
                      "train_f1_mac=", "{:.5f}".format(train_f1_mac), 
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_f1_mic=", "{:.5f}".format(val_f1_mic), 
                      "val_f1_mac=", "{:.5f}".format(val_f1_mac), 
                      "time=", "{:.5f}".format(avg_time))
 
            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
                break
    
    print("Optimization Finished!")
    sess.run(val_adj_info.op)
    val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
    print("Full validation stats:",
                  "loss=", "{:.5f}".format(val_cost),
                  "f1_micro=", "{:.5f}".format(val_f1_mic),
                  "f1_macro=", "{:.5f}".format(val_f1_mac),
                  "time=", "{:.5f}".format(duration))
    with open(log_dir() + "val_stats.txt", "w") as fp:
        fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}".
                format(val_cost, val_f1_mic, val_f1_mac, duration))

    print("Writing test set stats to file (don't peak!)")
    val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size, test=True)
    with open(log_dir() + "test_stats.txt", "w") as fp:
        fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f}".
                format(val_cost, val_f1_mic, val_f1_mac))


#added by Amine "chunks to train a model"

def train_chunks(train_data, test_data=None):
    start = time.time()
    Graphs = train_data[0]
    Graphs_train = train_data[1]
    Graphs_val = train_data[2]
    Graphs_test = train_data[3]
    #print(Graphs)
    #print(Graphs_val)
    #print(Graphs_test)
    #sys.exit(1)	
    features = train_data[4]
    id_maps = train_data[5]
    class_maps  = train_data[6]
    num_classes = train_data[7]
    #scaler = train_data[8]
    #print(id_maps)
    #print(class_maps)
    #sys.exit(1)	
    #if isinstance(list(class_map.values())[0], list):
        #num_classes = len(list(class_map.values())[0])
    #else:
        #num_classes = len(set(class_map.values()))
    for item in features.keys():	
        if not features is None:
            # pad with dummy zero vector
            features[item] = np.vstack([features[item], np.zeros((features[item].shape[1],))])

    #context_pairs = train_data[3] if FLAGS.random_context else None
    #placeholders={}
    #cpt=0
    adj_info_phs={}
    adj_infos={}	
    placeholders = construct_placeholders(num_classes)
    minibatch = GraphMinibatchIterator(Graphs, Graphs_train, Graphs_val, Graphs_test, 
            id_maps,
            placeholders, 
            class_maps,
            num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree)

    #adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    #adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
	
    # for g in minibatch.train_val_graphs:
        # #iter = 0
        # #training_round =0
        # if cpt<minibatch.train_instances:    			
            # adj_info_phs[g.name] = tf.placeholder(tf.int32, shape=minibatch.adj[g.name].shape)
            # adj_infos[g.name] = tf.Variable(adj_info_phs[g.name], trainable=False, name="adj_info")
            # #placeholders[g.name] = construct_placeholders(num_classes)			
            # cpt=cpt+1
        # else:
            # break

    for g in minibatch.train_val_graphs:
        #iter = 0
        #training_round =0
        #print(g.name)		
        adj_info_phs[g.name] = tf.placeholder(tf.int32, shape=minibatch.adj[g.name].shape)
        adj_infos[g.name] = tf.Variable(adj_info_phs[g.name], trainable=False, name="adj_info")
        #placeholders[g.name] = construct_placeholders(num_classes)
    #print("test")		
    for g in minibatch.test_graphs:
        #print(minibatch.test_adj[g.name].shape)
        adj_info_phs[g.name] = tf.placeholder(tf.int32, shape=minibatch.test_adj[g.name].shape)
        adj_infos[g.name] = tf.Variable(adj_info_phs[g.name], trainable=False, name="adj_info")			
			
    #minibatch.placeholders=placeholders
   # print(minibatch.placeholders)
    #print(placeholders)	
    #sys.exit(1)
    model = None	
    if FLAGS.model == 'graphsage_mean':
        # Create model
        #sampler = UniformNeighborSampler(adj_info)
        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", None, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", None, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", None, FLAGS.samples_3, FLAGS.dim_2)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", None, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", None, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", None, FLAGS.samples_1, FLAGS.dim_1)]

        model = SupervisedGraphsagePlus(num_classes, placeholders, 
                                     features,
                                     adj_infos,
                                     minibatch.deg,
                                     layer_infos, 
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    elif FLAGS.model == 'gcn':
        # Create model
        #sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", None, FLAGS.samples_1, 2*FLAGS.dim_1),
                            SAGEInfo("node", None, FLAGS.samples_2, 2*FLAGS.dim_2)]

        model = SupervisedGraphsagePlus(num_classes, placeholders, 
                                     features,
                                     adj_infos,
                                     minibatch.deg,
                                     layer_infos=layer_infos,
                                     aggregator_type="gcn",
                                     model_size=FLAGS.model_size,
                                     concat=False,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_seq':
        #sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", None, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", None, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsagePlus(num_classes, placeholders, 
                                     features,
                                     adj_infos,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="seq",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        #sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", None, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", None, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsagePlus(num_classes, placeholders, 
                                    features,
                                    adj_infos,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="maxpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_meanpool':
        #sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", None, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", None, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsagePlus(num_classes, placeholders, 
                                    features,
                                    adj_infos,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="meanpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True
    
    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
    #cpt=0
    feed={}
    #print(minibatch.adj.keys())
    # for g in minibatch.train_val_graphs:
        # #iter = 0
        # #training_round =0
        # if cpt<minibatch.train_instances:    					
            # feed[adj_info_phs[g.name]]=minibatch.adj[g.name]
            # cpt=cpt+1			
        # else:
            # break
    for g in minibatch.train_val_graphs:
        #iter = 0
        #training_round =0		
        feed[adj_info_phs[g.name]]=minibatch.adj[g.name]

    for g in minibatch.test_graphs:
        #iter = 0
        #training_round =0		
        feed[adj_info_phs[g.name]]=minibatch.test_adj[g.name]		

    saver = tf.train.Saver()
    #sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})	
    sess.run(tf.global_variables_initializer(), feed_dict=feed)
    # Train model   
    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []
    avg_time=[0]
    #train_adj_info = tf.assign(adj_info, minibatch.adj)
    #val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    #print(minibatch.train_instances)
    #print(len(minibatch.train_val_graphs))
    #print(placeholders)	
    for epoch in range(FLAGS.epochs):

        cpt=0
        train_cost=[]
        train_f1_mic=[]
        train_f1_mac=[]
        #avg_time=[0]
        val_cost=[]
        val_f1_mic=[]
        val_f1_mac=[]
        duration=[]
        f=True	
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        cpt=0		
        for g in minibatch.train_val_graphs:
            #iter = 0
            #training_round =0
		
            if cpt<minibatch.train_instances:
                #print(g.name)
                #sys.exit(1)				
                #print("Training Dependancies: "+g.name)
                minibatch.shuffle(g.name)
                minibatch.batch_num=0				
                while not minibatch.end(g.name):
                    feed_dict, labels = minibatch.next_minibatch_feed_dict(g.name)
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                    t = time.time()
                    #print(model)					
                    #print(merged)
                    #print(model.opt_op)
                    #print(model.loss)
                    #print(model.preds[g.name])
                    #print(feed_dict)
                    #print(labels)					
                    #input()				
                    # Training step per training graph
                    #print(sess.run([merged, model.opt_op, model.loss, model.preds[g.name]], feed_dict=feed_dict))
                    #sys.exit(1)					
                    #outs = sess.run([merged, model.opt_op, model.loss, model.preds[g.name]], feed_dict=feed_dict)
                    outs = sess.run([merged, model.opt_op, model.loss, model.preds], feed_dict=feed_dict)					
                    #train_cost = outs[2]
                     					
                    train_cost.append(outs[2])					
                    #train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1])
                    mic, mac = calc_f1(labels, outs[-1])					
                    train_f1_mic.append(mic)
                    train_f1_mac.append(mac)					
                    #avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)
                    avg_time.append((avg_time[-1] * total_steps + time.time() - t) / (total_steps + 1))					
                    #if total_steps % FLAGS.print_every == 0:
                    summary_writer.add_summary(outs[0], total_steps)
                    #print(#"iter=", '%04d' % iter, 
                          #"train_loss=", "{:.5f}".format(train_cost),
                          #"train_f1_mic=", "{:.5f}".format(train_f1_mic), 
                          #"train_f1_mac=", "{:.5f}".format(train_f1_mac), 
                          #"time=", "{:.5f}".format(avg_time))
                        #iter += 1
                    total_steps += 1

                    if total_steps > FLAGS.max_total_steps:
                        break
						
                cpt=cpt+1					
            else:
                #sys.exit(1)			
                #if iter % FLAGS.validate_iter == 0:
                    # Validation
                f=False				
                #print("Validation Dependancies: "+g.name)					
                train_adj_info = tf.assign(adj_infos[g.name], minibatch.adj[g.name])
                #val_adj_info = tf.assign(adj_infos[g.name], minibatch.test_adj[g.name])		
                sess.run(train_adj_info.op)
                if FLAGS.validate_batch_size == -1:
                    #val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate_graph(sess, model, minibatch, g.name, FLAGS.batch_size)
                    val_cost_v, val_f1_mic_v, val_f1_mac_v, duration_v = incremental_evaluate_graph(sess, model, minibatch, g.name, FLAGS.batch_size)					
                else:
                    #val_cost, val_f1_mic, val_f1_mac, duration = evaluate_graph(sess, model, minibatch, g.name, FLAGS.validate_batch_size)
                    val_cost_v, val_f1_mic_v, val_f1_mac_v, duration_v = evaluate_graph(sess, model, minibatch, g.name, FLAGS.validate_batch_size)					
                #sess.run(train_adj_info.op)
                #epoch_val_costs[-1] += val_cost
                epoch_val_costs[-1] += val_cost_v
                val_cost.append(val_cost_v)
                val_f1_mic.append(val_f1_mic_v)
                val_f1_mac.append(val_f1_mac_v)
                duration.append(duration_v)				
                #print(#"Iter:", '%04d' % iter, 
                      #"val_loss=", "{:.5f}".format(val_cost),
                      #"val_f1_mic=", "{:.5f}".format(val_f1_mic), 
                      #"val_f1_mac=", "{:.5f}".format(val_f1_mac), 
                      #"time=", "{:.5f}".format(duration))
        #if total_steps > FLAGS.max_total_steps:
            #break

        print(#"iter=", '%04d' % iter, 
                "train_loss=", "{:.5f}".format(sum(train_cost)/len(train_cost)),
                "train_f1_mic=", "{:.5f}".format(sum(train_f1_mic)/len(train_f1_mic)), 
                "train_f1_mac=", "{:.5f}".format(sum(train_f1_mac)/len(train_f1_mac)), 
                "time=", "{:.5f}".format(sum(avg_time)/len(avg_time)))

        print(#"Iter:", '%04d' % iter, 
                "val_loss=", "{:.5f}".format(sum(val_cost)/len(val_cost)),
                "val_f1_mic=", "{:.5f}".format(sum(val_f1_mic)/len(val_f1_mic)), 
                "val_f1_mac=", "{:.5f}".format(sum(val_f1_mac)/len(val_f1_mac)), 
                "time=", "{:.5f}".format(sum(duration)/len(duration)))
				

			
    print("Optimization Finished!")
    print("Starting Testing")
    #sess.run(val_adj_info.op)
    #sess.run(train_adj_info.op)	
    res={'name':[],'cost':[],'f1_mic':[], 'f1_mac':[], 'duration':[]}

    cpt=0	
    for g in minibatch.test_graphs:
        adj_info = tf.assign(adj_infos[g.name], minibatch.test_adj[g.name])	
        sess.run(adj_info.op)
        if FLAGS.validate_batch_size == -1:		
            val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate_graph(sess, model, minibatch, g.name, FLAGS.batch_size, test=True)
        else:
            val_cost, val_f1_mic, val_f1_mac, duration = evaluate_graph(sess, model, minibatch, g.name, FLAGS.validate_batch_size, test=True)		
        res['name'].append(g.name)		
        res['cost'].append(val_cost)
        res['f1_mic'].append(val_f1_mic)
        res['f1_mac'].append(val_f1_mac)
        res['duration'].append(duration)
        cpt = cpt+1		
    avgs={}
    with open(log_dir() + "test_stats.txt", "w") as fp:
        for k,v in res.items():
            if not isinstance(v[0], str):		
                avgs[k] = sum(v)/ float(len(v))
		
        fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} avg_time={:.5f}".format(avgs['cost'], avgs["f1_mic"], avgs["f1_mac"], avgs["duration"]))
        fp.write("\n")	
        for i in range(0,cpt):
            fp.write("name={:s} loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} avg_time={:.5f}".format(res['name'][i], res['cost'][i], res["f1_mic"][i], res["f1_mac"][i], res["duration"][i]))
            fp.write("\n")       		
    print("time (Building+Training+Validation+Testing) =", "{:.5f}".format(time.time()-start))
    print('Saving model...')
    #print(sess.graph.get_collection())
    #sys.exit(1)	
    saver.save(sess, log_dir()+"model.cpkt")
    #with open(log_dir()+"gsage.pkl","wb") as f:	
        #pickle.dump(model, f)
    #joblib.dump(model, log_dir()+"gsage.pkl")		
    #num classes, max degree, max nodes, validate batch size, batch size		
    params={0:num_classes, 1:FLAGS.max_degree, 2: FLAGS.nodes_max, 3: FLAGS.validate_batch_size, 4: FLAGS.batch_size}
    with open(log_dir()+"params.pkl","wb") as f:	
        pickle.dump(params, f)		
    # if scaler != None:
        # with open(log_dir()+"scaler.pkl","wb") as f:
            # pickle.dump(scaler, f)
    # if model != None:
        # with open(log_dir()+"gsage.pkl","wb") as f:
            # pickle.dump(model, f)

def test_graph(name,sess,infos,model,t_test): #test data needs to contain adj_infos[g.name], adj[g.name] 
    adj_info = tf.assign(infos[0], infos[1])
    sess.run(adj_info.op)
    #t_test = time.time()
    #print(minibatch_iter.val_nodes)
    #sys.exit(1)
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(graph_id,size)
    node_outs_val = sess.run([model.preds, model.loss], 
                        feed_dict=infos[2])
    mic, mac = calc_f1(infos[3], node_outs_val[0])
    return node_outs_val[1], mic, mac, (time.time() - t_test), infos[3], node_outs_val[0]

def test_graph_incremental(name,sess,infos,model,t_test): #test data needs to contain adj_infos[g.name], adj[g.name] 
    adj_info = tf.assign(infos[0], infos[1])
    sess.run(adj_info.op)
    #t_test = time.time()
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    for i in range(len(infos[3].keys())):
        node_outs_val = sess.run([model.preds, model.loss], 
                         feed_dict=infos[3][i][0])
        val_preds.append(node_outs_val[0])
        labels.append(infos[3][i][1])
        val_losses.append(node_outs_val[1])
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test), labels, vals_preds
    	


def batch_feed_dict(batch_nodes, label_map, num_classes, val=False):
		
    batch1 = [self.id2idx[str(n)] for n in batch_nodes]		
    labels = np.vstack([make_label_vec(node,label_map, num_classes) for node in batch_nodes])
    feed_dict = dict()
    feed_dict.update({placeholders['batch_size'] : len(batch1)})
    feed_dict.update({placeholders['batch']: batch1})
    feed_dict.update({placeholders['labels']: labels})

    return feed_dict, labels
				
def make_label_vec(node,label_map,num_classes):
    label = label_map[str(node)]
    if isinstance(label, list):
        label_vec = np.array(label)
    else:
        label_vec = np.zeros((num_classes))
        class_ind = label_map[str(node)]
        label_vec[class_ind] = 1
    return label_vec
	
def node_val_feed_dict(G, label_map, num_classes, size=None): #id is graph id

    val_nodes = [n for n in G.nodes()]	
    if not size is None:
        val_nodes = np.random.choice(val_nodes, size, replace=True)
    # add a dummy neighbor
    ret_val = batch_feed_dict(val_nodes, label_map, num_classes)
    return ret_val[0], ret_val[1]

def incremental_node_val_feed_dict(G, size, iter_num, label_map, num_classes, test=False): #id is graph id
    val_nodes = [n for n in G.nodes()]				
    val_node_subset = val_nodes[iter_num*size:min((iter_num+1)*size, len(val_nodes))]
    # add a dummy neighbor
    ret_val = batch_feed_dict(val_node_subset, label_map, num_classes)
    return ret_val[0], ret_val[1], (iter_num+1)*size >= len(val_nodes), val_node_subset

def build_input_incremental(data, max_degree, batch_size, num_classes):
    G = data[0]
    #feats = data[1]
    id_map = data[2]
    class_map =data[3]
    adj=construct_test_adj(G,id_map,max_degree)
    iter_num = 0
    feeds={}
    while not finished:		
        feed_dict_val, batch_labels, finished, _  = incremental_node_val_feed_dict(G, batch_size, iter_num, class_map, num_classes)
        feeds[iter_num]=(feed_dict_val, batch_labels)
        iter_num += 1
    adj_info_ph = tf.placeholder(tf.int32, shape=adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
    return adj_info,adj,feeds

def build_input(data, max_degree, batch_size, num_classes):
    G = data[0]
    #feats = data[1]
    id_map = data[2]
    class_map =data[3]
    adj=construct_test_adj(G,id_map,max_degree)
    feed_dict_val, labels = node_val_feed_dict(G,batch_size,num_classes)
    adj_info_ph = tf.placeholder(tf.int32, shape=adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
    return adj_info,adj, feed_dict_val, labels
	
def construct_test_adj(G,id2idx,max_degree):
    adj = len(id2idx)*np.ones((len(id2idx)+1, max_degree))
    for nodeid in G.nodes():
        neighbors = np.array([id2idx[str(neighbor)] 
            for neighbor in G.neighbors(nodeid)])
        if len(neighbors) == 0:
            continue
        if len(neighbors) > max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)
        elif len(neighbors) < max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=True)
        adj[id2idx[str(nodeid)], :] = neighbors
    return adj
	
def main(argv=None):
    # if tf.test.gpu_device_name():
        # print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    # else:
        # print("Please install GPU version of TF")
    # input()
    if FLAGS.train_mode==True:
        if FLAGS.train_chunks!=True:
            print("Loading training data..")
            #train_data = load_data(FLAGS.train_prefix,FLAGS.train_on_val_test)
            train_data = load_data(FLAGS.train_prefix)
            print("Done loading training data..")
            train(train_data)
        else:
            print("Loading training data..")
            #Number of chunks important to catch the maximum number of graph samples 
            train_data= load_data_chunks(FLAGS.train_prefix, log_dir()+"scaler.pkl", FLAGS.data_prefix, FLAGS.train_percentage, FLAGS.nodes_max)
            print("Done loading chunked training data..")
            train_chunks(train_data)
    else: #we load a model and a scaler from a path to test additional data	
        print("Loading model for additional testing")
        sess=tf.Session()		
        saver = tf.train.import_meta_graph(FLAGS.model_dir)
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        scaler = None
        #model = None
        params = None		
        #with open(FLAGS.model_dir+os.sep+"gsage.pkl","rb") as f:		
            #gsage =  pickle.load(f)
        #with open(FLAGS.model_dir+os.sep+"scaler.pkl","rb") as f:
            #scaler = pickle.load(f)
        #path = FLAGS.model_dir+os.sep+"gsage.pkl"
        #if os.path.exists(path):
            #gsage = joblib.load(path)		
        path = FLAGS.model_dir+os.sep+"scaler.pkl"
        if os.path.exists(path):
            scaler = joblib.load(path)
        with open(FLAGS.model_dir+os.sep+"params.pkl","rb") as f: # parameters as a dictionary, num classes, max degree, max nodes, validate batch size, batch size
            params =  pickle.load(f)		
        tests = [os.path.join(FLAGS.test_dir, f) for f in os.listdir(FLAGS.test_dir) if os.path.isdir(os.path.join(FLAGS.test_dir, f))]
        for item in tests:		
            test_data = load_graph_test(scaler, item, FLAGS.data_prefix, params[2])#graph, feats, id_map_file, class_map
            t_test = time.time()			
            if params[3]==-1:
                #max_degree, batch_size, num_classes			
                infos = build_input_incremental(test_data,params[1],params[4],params[0])
                loss, mic, mac, duration, label, preds=test_graph_incremental(test_data[0], sess, infos, gsage,t_test)
                print(item)				
                print("Results:",
                  "loss=", "{:.5f}".format(loss),
                  "f1_micro=", "{:.5f}".format(mic),
                  "f1_macro=", "{:.5f}".format(mac),
                  "time=", "{:.5f}".format(duration))				
            else:
                infos = build_input_incremental(test_data,params[1],params[3],params[0])			
                loss, mic, mac, duration, label, preds=test_graph(test_data[0].name, sess, infos, gsage,t_test)
                print(item)				
                print("Results:",
                  "loss=", "{:.5f}".format(loss),
                  "f1_micro=", "{:.5f}".format(mic),
                  "f1_macro=", "{:.5f}".format(mac),
                  "time=", "{:.5f}".format(duration))				
				
if __name__ == '__main__':
    tf.app.run()
