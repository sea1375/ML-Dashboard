import tensorflow as tf
import sys

import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator
from graphsage.neigh_samplers import UniformNeighborSampler

flags = tf.app.flags
FLAGS = flags.FLAGS

class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            placeholders, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean", 
            model_size="small", sigmoid_loss=False, identity_dim=0,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
		
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
			
        else:
            #print(features.shape)		
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            #print(self.features)
            #print(self.embeds)
            #input()			
            if not self.embeds is None:

                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        #print(self.features.shape)
        self.build()


    def build(self):
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                support_sizes1, concat=self.concat, model_size=self.model_size)
        dim_mult = 2 if self.concat else 1

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)

        dim_mult = 2 if self.concat else 1
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes, 
                dropout=self.placeholders['dropout'],
                act=lambda x : x)
        # TF graph management
        self.node_preds = self.node_pred(self.outputs1)

        self._loss()
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        #print(clipped_grads_and_vars)		
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        #print(self.opt_op)		
        self.preds = self.predict()

    def _loss(self):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
       
        # classification loss
        if self.sigmoid_loss:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))
        else:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))

        tf.summary.scalar('loss', self.loss)

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)


class SupervisedGraphsagePlus(models.SampleAndAggregatePlus):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            placeholders, features, adjs, degrees,
            layer_infos, concat=True, aggregator_type="mean", 
            model_size="small", sigmoid_loss=False, identity_dim=0,
                **kwargs):
        '''
        Args:
            - placeholders: indexed Stanford TensorFlow placeholder object.
            - features: indexed Numpy arrays with graphs features.
            - adjs: indexed dictionary containing Numpy array with adjacency lists (padded with random re-samples)
            - degrees: indexed dicitionary containing Numpy array with node degrees. 
            - layer_infos: indexed List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)
        self.ids = adjs.keys()
        # get info from placeholders...
        #self.inputs1={}	
        #self.inputs2={}
        self.embeds={}
        self.features={}
        self.dims={}
        self.samplers={}		
        #for item in self.ids:		
            #self.inputs1[item] = placeholders["batch"]
            #self.inputs2[item] = placeholders[item]["batch2"]
        self.inputs1 = placeholders["batch"]			
        self.model_size = model_size
        self.adj_info = adjs
        for item in self.ids:
            self.samplers[item]= UniformNeighborSampler(adjs[item])		

        if identity_dim > 0:
           #self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
           for item in self.ids:		
               self.embeds[item] = tf.get_variable("node_embeddings", [adjs[item].get_shape().as_list()[0], identity_dim])	   
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            for item in features.keys():		
                self.features[item] = tf.Variable(tf.constant(features[item], dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features[item] = tf.concat([self.embeds[item], self.features[item]], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        #self.dims[item] = [(0 if features is None else features.shape[1]) + identity_dim[item]]
        for item in features.keys():
            #print(identity_dim[item])			
            self.dims[item] = [(0 if features[item] is None else features[item].shape[1]) + identity_dim]
            #self.dims[item] = [features[item].shape[1] + identity_dim[item]]			
            #self.dims[item].extend([layer_infos[item][i].output_dim for i in range(len(layer_infos[item]))])
            self.dims[item].extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        #self.batch_size = placeholders[list(features.keys())[0]]["batch_size"] #not clean a little hack
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def build(self):
        #self.outputs1={}
        #self.node_pred={}
        self.aggregators={}
        self.node_preds={}
        #self.preds={}
        #print("I am in build")		
        #sys.exit(1)
        #output dense layer outside a loop considering potential max dimension in feeding tensors		
        dim_mult = 2 if self.concat else 1
        self.node_pred = layers.Dense(dim_mult*max(item[1][-1] for item in self.dims.items()), self.num_classes, 
            dropout=self.placeholders['dropout'],
            act=lambda x : x)
	
        for item in self.ids:
            #print(self.layer_infos)
            #print(self.inputs1[item])
            #print(self.samplers[item])
            #print(self.dims[item])
            #input()			
            #sys.exit(1)			
            #samples1, support_sizes1 = self.sample(self.inputs1[item], self.layer_infos, self.samplers[item])
            samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos, self.samplers[item])			
            num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
            #self.outputs1[item], self.aggregators[item] = self.aggregate(item, samples1, [self.features[item]], self.dims[item], num_samples,
                #support_sizes1, concat=self.concat, model_size=self.model_size)
            #print(self.features[item])
            #print(self.dims[item])
            #input()					
            #self.outputs1, self.aggregators = self.aggregate(samples1, [self.features[item]], self.dims[item], num_samples,
                #support_sizes1, concat=self.concat, model_size=self.model_size)
            #print(self.features[item])
            #print(samples1)
            #input()			
            self.outputs1, self.aggregators[item] = self.aggregate(samples1, [self.features[item]], self.dims[item], num_samples,
                support_sizes1, concat=self.concat, model_size=self.model_size)					
            #dim_mult = 2 if self.concat else 1
            #print(self.outputs1)
            #print(self.aggregators)
            #input()			
            #self.outputs1[item] = tf.nn.l2_normalize(self.outputs1[item], 1)
            self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
            #dim_mult = 2 if self.concat else 1
            # self.node_pred[item] = layers.Dense(dim_mult*self.dims[item][-1], self.num_classes, 
                # dropout=self.placeholders['dropout'],
                # act=lambda x : x)

				
            # TF graph management
            #self.node_preds[item] = self.node_pred[item](self.outputs1[item])
            #print(self.outputs1)			
            #self.node_preds = self.node_pred(self.outputs1)
            self.node_preds[item] =self.node_pred(self.outputs1) 			
            #print(self.node_preds)
            #input()			
        self._loss()
		
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        #print(grads_and_vars)			
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
            for grad, var in grads_and_vars]
        #for i in range(len(clipped_grads_and_vars)):				
            #print(clipped_grads_and_vars[i])
            #input()			
        self.grad, _ = clipped_grads_and_vars[0]
        #print(self.grad)		
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        #print(self.opt_op)		
        #for item in self.ids:		
            #self.preds[item] = self.predict(item)
        self.preds = self.predict()
        #print("Exiting build")			


    def _loss(self):
        tmp=0.0	
        # Weight decay loss
        for item in self.ids:		
            for aggregator in self.aggregators[item]:
                for var in aggregator.vars.values():
                    #self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
                    tmp += FLAGS.weight_decay * tf.nn.l2_loss(var)					
        tmp=tmp/len(self.ids)
        self.loss += tmp		
        for var in self.node_pred.vars.values():
            #self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)			
        for item in self.ids:      
            # classification loss
            if self.sigmoid_loss:
                # self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    # logits=self.node_preds[item],
                    # labels=self.placeholders['labels']))
                tmp += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds[item],
                    labels=self.placeholders['labels']))					
            else:
                # self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    # logits=self.node_preds[item],
                    # labels=self.placeholders['labels']))
                tmp += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds[item],
                    labels=self.placeholders['labels']))					
        tmp=tmp/len(self.ids)
        self.loss += tmp        
        tf.summary.scalar('loss', self.loss)


    def predict(self):
        if self.sigmoid_loss:
            vals=[tf.nn.sigmoid(self.node_preds[item]) for item in self.ids]            
            avg = sum(vals) / len(vals)
            return avg
        else:
            vals=[tf.nn.softmax(self.node_preds[item]) for item in self.ids]            
            avg = sum(vals) / len(vals)
            return avg
