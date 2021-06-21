# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/test_gnn.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# display(df)
df.cache()
author = df.select('author2').toDF('author').union(df.select('author1').toDF('author'))
# print(author.count())
author_distinct = author.dropDuplicates()
# print(author_distinct.count())
display(author_distinct)

# COMMAND ----------

print(author.count())
print(author_distinct.count())

# initial want to use join, but too large!

# COMMAND ----------

from pyspark.sql.functions import concat_ws
edge = df.withColumn('edge', concat_ws('_', df['author1'], df['author2']))
adj_left = edge.select('author1', 'author2', 'edge')
adj_right = edge.select('author1', 'author2', 'edge')
#adj = adj_left.join(adj_right, adj_left['author1'] == adj_left['author1'] | adj_left['author1'] == adj_right['author2'] | adj_right['author2'] == adj_right['author2'], 'inner' )
adj1 = adj_left.join(adj_right, ['author1'],'inner').toDF('author1', 'author2','edge1', 'author22','edge2')
print(adj1.count())


# COMMAND ----------

adj2 = adj_left.join(adj_right, ['author2'],'inner').toDF('author2', 'author1','edge1', 'author11','edge2')
print(adj2.count())
adj2.show()

# COMMAND ----------

adj1.show()

# COMMAND ----------

edge.show()

# COMMAND ----------

# let the edge bacome the node
# df.cache()
# load tabular data
edge.select('edge', 'common_neighbors', 'total_neighbors', 'preferential_attachment', 'jaccard', 'adamic_adar', 'resource_allocation', 'leicht_holme_nerman', 'sorensen_index', 'salton_cosine_similarity', 'hub_promoted', 'hub_depressed', 'target').toPandas()

# COMMAND ----------

adj_map = adj1.select('edge1', 'edge2').toPandas().to_numpy()

# COMMAND ----------

# spark dataframe 2 pandas dataframe 2 numpy dataframe
train_dataset = edge.select('edge', 'common_neighbors', 'total_neighbors', 'preferential_attachment', 'jaccard', 'adamic_adar', 'resource_allocation', 'leicht_holme_nerman', 'sorensen_index', 'salton_cosine_similarity', 'hub_promoted', 'hub_depressed', 'target').toPandas().to_numpy() # the data is string, change the straing to the double

# COMMAND ----------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# COMMAND ----------

import scipy.sparse as sp
import numpy as np

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(dataset, idx_features_labels, edges_unordered):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    #idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32) Python int too large to convert to C long
    idx = np.array(idx_features_labels[:, 0])
                   
    idx_map = {j: i for i, j in enumerate(idx)}
    #edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),dtype=np.int32) 
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(idx_features_labels.shape[0]-10000)
    idx_val = range(idx_features_labels.shape[0]-10000, idx_features_labels.shape[0])
    idx_test = range(idx_features_labels.shape[0]-10000, idx_features_labels.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# COMMAND ----------

from __future__ import division
from __future__ import print_function

import time
import argparse
import torch.optim as optim
from collections import namedtuple

# Training settings
Params = namedtuple('Params', ['batch_size', 'test_batch_size', 'epochs', 'lr', 'momentum', 'seed', 'cuda', 'log_interval', 'hidden', 'fastmode','weight_decay','dropout'])
args = Params(batch_size=200, test_batch_size=10000, epochs=10, lr=0.01, momentum=0.5, seed=1, cuda=False, log_interval=200, hidden=16, fastmode=False, weight_decay=5e-4, dropout=0.5)
torch.manual_seed(args.seed)

# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# load_data(dataset="cora", idx_features_labels, edges_unordered
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset = 'academical', idx_features_labels = train_dataset, edges_unordered = adj_map)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

# COMMAND ----------

import horovod.torch as hvd
from sparkdl import HorovodRunner
def train_hvd(learning_rate):
  
  # Initialize Horovod
  hvd.init()  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  if device.type == 'cuda':
    # Pin GPU to local rank
    torch.cuda.set_device(hvd.local_rank())
 
  train_dataset = datasets.MNIST(
    # Use different root directory for each worker to avoid conflicts
    root='data-%d'% hvd.rank(),  
    train=True, 
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
  )
 
  from torch.utils.data.distributed import DistributedSampler
  
  # Configure the sampler so that each worker gets a distinct sample of the input dataset
  train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
  # Use train_sampler to load a different sample of data on each worker
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
 
  model = Net().to(device)
  
  # The effective batch size in synchronous distributed training is scaled by the number of workers
  # Increase learning_rate to compensate for the increased batch size
  optimizer = optim.SGD(model.parameters(), lr=learning_rate * hvd.size(), momentum=momentum)
 
  # Wrap the local optimizer with hvd.DistributedOptimizer so that Horovod handles the distributed optimization
  optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
  
  # Broadcast initial parameters so all workers start with the same parameters
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)
 
  for epoch in range(1, num_epochs + 1):
    train_one_epoch(model, device, train_loader, optimizer, epoch)
    # Save checkpoints only on worker 0 to prevent conflicts between workers
    if hvd.rank() == 0:
      save_checkpoint(model, optimizer, epoch)

hr = HorovodRunner(np=2) 
hr.run(train_hvd, learning_rate = 0.001)

# COMMAND ----------

def eval():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    classnum = 2
    target_num = torch.zeros((1,classnum))
    predict_num = torch.zeros((1,classnum))
    acc_num = torch.zeros((1,classnum))
    
    optimizer.zero_grad()
    output = model(features, adj)
    
    outputs = output[idx_test]
    targets = labels[idx_test]
    loss = F.nll_loss(outputs, targets)
    
    test_loss += loss.data.data.item()
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

    pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
    predict_num += pre_mask.sum(0)
    tar_mask = torch.zeros(outputs.size()).scatter_(1, targets.data.cpu().view(-1, 1), 1.)
    target_num += tar_mask.sum(0)
    acc_mask = pre_mask*tar_mask
    acc_num += acc_mask.sum(0)
    #

    recall = acc_num/target_num
    precision = acc_num/predict_num
    F1 = 2*recall*precision/(recall+precision)
    accuracy = acc_num.sum(1)/target_num.sum(1)

    recall = (recall.numpy()[0]*100).round(3)
    precision = (precision.numpy()[0]*100).round(3)
    F1 = (F1.numpy()[0]*100).round(3)
    accuracy = (accuracy.numpy()[0]*100).round(3)

    print('recall'," ".join('%s' % id for id in recall))
    print('precision'," ".join('%s' % id for id in precision))
    print('F1'," ".join('%s' % id for id in F1))
    print('accuracy',accuracy)
    
eval()

# COMMAND ----------


